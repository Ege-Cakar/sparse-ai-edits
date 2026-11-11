"""Fully AI-generated counterparts of the calib and test documents, the null of the second-stage test H0' (every
sentence AI; stage2.py). Decided 2026-11-04 by Ege.

--topics: generate.topic_writer (an edit.editors entry) reads each human document of generate.splits, rebuilt as the
editors saw it (edit.documents), and writes a one-line topic (TOPIC).
--generators NAME...: each generator (an edit.editors entry, with its model, product and sampling settings, the
output cap raised to generate.max_tokens) writes a document of the source's genre (GENRE) on that topic, about as many
sentences long as the human document (ASK); it never sees the human text. Documents without a topic yet are skipped.
Both steps reject an empty reply or one cut off at the output cap (and a topic of more than one line), resample up to
generate.max_attempts times, and append to generated/topics.jsonl and generated/<generator>.jsonl (doc_id, text,
attempts, rejected), skipping finished documents, so they can be rerun after an interruption. Usage goes to
<file>.usage as in edit.py; --max-usd caps a step by it. Anthropic models run through the Message Batches API (batch
state in <file>.batches, collected on the next run, as edit.py --batch); the others send concurrent chat completions.
--pilot K limits a step to the first K documents of each source in one seeded order; its records are kept, so a
later full run skips them.
--segment: the generated documents' sentences, split as the corpus was (corpus.segmenter, segment.py; replies cached
in generated/segment_cache.jsonl) with the spacing kept, since generated text has no processing errors. Writes
generated/sentences.parquet and generated/versions.parquet: each sentence as a version of pseudo-editor
"generated-<generator>" (doc_id: the human document it mirrors; version_id and sentence_id from VERSION_OFFSET on, so
they never collide with versions.parquet), the input of detect.py --score --versions generated/versions.parquet.
"""
import asyncio
import json
import random
from pathlib import Path

import anthropic
import openai
import pandas as pd
from ftfy import fix_text
from tqdm import tqdm

from . import cli, path
from .corpus import FTFY
from .edit import SYSTEM, VERSION_COLS, client_args, documents, read_jsonl, results, spent, submit, \
    terminate_last_line
from .segment import DESCRIPTION, Segmenter
from .text import paragraphs, words

TOPIC = ("Text: {document}\n\nThis is {description}. Write its topic in one line of at most 20 words, without copying "
         "its wording. Reply with only the topic.")
GENRE = {"persuade": "a student essay", "ellipse": "a student essay", "enron": "an email",
         "fineweb_edu": "an educational web page", "openreview": "a peer review of a research paper",
         "pes2o": "a scientific paper", "writingprompts": "a short story", "xsum": "a news article"}
ASK = ("Write {genre} on this topic: {topic}\n\nIt should be about {n} sentences long. Reply with only the text, in "
       "plain text without Markdown formatting.")
CAPS = ("max_tokens", "max_completion_tokens")   # the output cap's name differs by API
VERSION_OFFSET = 10 ** 8


def body(model, prompt, max_tokens):
    """Request body for one prompt, with the model's system prompt and params, the output cap set to max_tokens."""
    params = {k: max_tokens if k in CAPS else v for k, v in model.get("params", {}).items()}
    system = SYSTEM.format(product=model["product"])
    if model.get("api") == "anthropic":
        return dict(model=model["model"], system=system, messages=[{"role": "user", "content": prompt}], **params)
    return dict(model=model["model"], messages=[{"role": "system", "content": system},
                                                {"role": "user", "content": prompt}], **params)


def rejection(text, usage, max_tokens, one_line=False):
    out = usage.get("output_tokens", usage.get("completion_tokens"))   # Anthropic, OpenAI
    if out is not None and out >= max_tokens:
        return "truncated"
    if not text.strip():
        return "empty"
    return "not_one_line" if one_line and len(text.strip().splitlines()) > 1 else None


def pilot_docs(cfg, docs, k):
    """The first k documents of each source in one seeded order (docs: one row per document, with source)."""
    return set(docs.sample(frac=1, random_state=cfg["seed"]).groupby("source").head(k).doc_id)


async def ask(client, request, cid, usage_file):
    """(reply text, token usage) of one chat completion; a request rate-limited or without flex capacity after the
    client's retries waits and is resent (as edit.edit_one); None on other API errors (retried next run)."""
    while True:
        try:
            resp = await client.chat.completions.create(**request)
            break
        except openai.APIError as e:
            if e.code in ("rate_limit_exceeded", "flex_unavailable"):
                await asyncio.sleep(random.uniform(10, 30))
                continue
            print(f"{cid}: {e!r}")
            return None
    usage = resp.usage.model_dump() if resp.usage else {}
    usage_file.write(json.dumps(dict(custom_id=cid, service_tier=resp.service_tier, usage=usage)) + "\n")
    return resp.choices[0].message.content or "", usage


async def run_concurrent(model, prompts, check, out_path, max_attempts, concurrency, max_tokens):
    client = openai.AsyncOpenAI(**client_args(model), timeout=1800)   # long generations under flex
    sem = asyncio.Semaphore(model.get("concurrency", concurrency))

    async def one(doc, u):
        async with sem:
            for attempt in range(1, max_attempts + 1):
                if (r := await ask(client, body(model, prompts[doc], max_tokens), f"{doc}_{attempt}", u)) is None:
                    return None
                if (reason := check(*r)) is None:
                    break
            return dict(doc_id=doc, text=r[0].strip(), attempts=attempt, rejected=reason)

    terminate_last_line(out_path)
    failed = 0
    with open(out_path, "a", buffering=1) as f, open(f"{out_path}.usage", "a", buffering=1) as u, \
            tqdm(total=len(prompts), desc=out_path.stem) as bar:
        for fut in asyncio.as_completed([one(doc, u) for doc in prompts]):
            res = await fut
            failed += res is None
            if res is not None:
                f.write(json.dumps(res) + "\n")
            bar.update()
    print(f"{out_path.stem}: {failed} requests failed after retries; rerun to retry them")


def run_batches(model, prompts, check, out_path, max_attempts, max_tokens, max_usd=None):
    """One Message Batch of the documents still to do (Anthropic allows 100,000 requests and 256 MB per batch), then
    one per round of resamples; batches already in the state file are collected first."""
    state = Path(f"{out_path}.batches")
    client = anthropic.Anthropic(api_key=client_args(model)["api_key"], max_retries=8)
    jobs = [(doc, 1) for doc in prompts]
    while jobs or state.exists():
        if jobs:
            cost, n = spent(model, out_path) if max_usd else (0.0, 0)
            if max_usd and (not n or cost + cost / n * len(jobs) > max_usd):
                raise SystemExit(f"{out_path.stem}: ${cost:.2f} spent; the next {len(jobs)} requests would pass "
                                 f"${max_usd} (or no usage measured yet)")
            group = [(f"{doc}_{attempt}", body(model, prompts[doc], max_tokens)) for doc, attempt in jobs]
            state.write_text(json.dumps((json.loads(state.read_text()) if state.exists() else [])
                                        + [submit(client, model, group)]))
        jobs = []
        terminate_last_line(out_path)
        with open(out_path, "a", buffering=1) as f, open(f"{out_path}.usage", "a") as u:
            for batch_id in json.loads(state.read_text()):
                for cid, text, usage in results(client, model, batch_id):
                    u.write(json.dumps(dict(custom_id=cid, batch_id=batch_id, usage=usage)) + "\n")
                    doc, attempt = map(int, cid.split("_"))
                    if (reason := check(text, usage)) and attempt < max_attempts:
                        jobs.append((doc, attempt + 1))
                    else:
                        f.write(json.dumps(dict(doc_id=doc, text=text.strip(), attempts=attempt, rejected=reason))
                                + "\n")
        state.unlink()


def run_step(cfg, model, prompts, out_path, one_line, max_usd):
    """Send prompts {doc_id: prompt} not yet answered in out_path, through the model's API."""
    g = cfg["generate"]
    done = {r["doc_id"] for r in read_jsonl(out_path)}
    todo = {doc: p for doc, p in prompts.items() if doc not in done}
    print(f"{out_path.stem}: {len(todo)} documents to send")
    if model.get("api") == "anthropic":
        if Path(f"{out_path}.batches").exists() or todo:
            run_batches(model, todo, lambda t, u: rejection(t, u, g["max_tokens"], one_line), out_path,
                        g["max_attempts"], g["max_tokens"], max_usd)
    elif todo:
        if max_usd:
            cost, n = spent(model, out_path)
            if not n or cost + cost / n * len(todo) > max_usd:
                raise SystemExit(f"{out_path.stem}: ${cost:.2f} spent; {len(todo)} more requests would pass "
                                 f"${max_usd} (or no usage measured yet)")
        asyncio.run(run_concurrent(model, todo, lambda t, u: rejection(t, u, g["max_tokens"], one_line), out_path,
                                   g["max_attempts"], g["concurrency"], g["max_tokens"]))


def accepted(path_):
    """Last record per document whose text was accepted."""
    rows = pd.DataFrame(read_jsonl(path_), columns=["doc_id", "text", "attempts", "rejected"])
    rows = rows.drop_duplicates("doc_id", keep="last")
    return rows[rows.rejected.isna()]


def segment(cfg, sents, generators):
    """generated/sentences.parquet and generated/versions.parquet from every generator's accepted documents."""
    seg = cfg["corpus"]["segmenter"] | {"fix_spacing": [], "rejoin_hyphens": []}
    segmenter, meta = Segmenter(seg, path(cfg, "generated", "segment_cache.jsonl")), \
        sents.drop_duplicates("doc_id").set_index("doc_id")[["source", "split"]]
    rows = []
    for gen in generators:
        docs = accepted(path(cfg, "generated", f"{gen}.jsonl")).join(meta, on="doc_id").sort_values("doc_id")
        for source, d in docs.groupby("source"):
            for i in range(0, len(d), seg["batch_docs"]):
                part = d.iloc[i:i + seg["batch_docs"]]
                paras = [paragraphs(fix_text(t, FTFY), "\n") for t in part.text]   # every line break: a paragraph
                for doc, split, out in zip(part.doc_id, part.split, segmenter(source, "\n", paras)):
                    rows += [dict(generator=gen, source=source, doc_id=doc, split=split, sent_idx=k, para_idx=p,
                                  text=s, n_words=len(words(s)), segmenter=how) for k, (p, s, how) in enumerate(out)]
    df = pd.DataFrame(rows)
    df.insert(0, "sentence_id", VERSION_OFFSET + pd.RangeIndex(len(df)))
    df.to_parquet(path(cfg, "generated", "sentences.parquet"), index=False)
    v = df.assign(version_id=df.sentence_id, editor="generated-" + df.generator, target_r=float("nan"), rep=0,
                  word_edits=float("nan"), actual_r=float("nan"))
    v[["version_id"] + VERSION_COLS].to_parquet(path(cfg, "generated", "versions.parquet"), index=False)
    print(df.groupby(["generator", "source"]).agg(docs=("doc_id", "nunique"), sentences=("text", "size")))


def main():
    def add_args(p):
        p.add_argument("--topics", action="store_true", help="write the topics with generate.topic_writer")
        p.add_argument("--generators", nargs="*", default=[], help="write these generators' documents")
        p.add_argument("--segment", nargs="*", metavar="GENERATOR", help="segment these generators' documents")
        p.add_argument("--pilot", type=int, metavar="K", help="only the first K documents of each source")
        p.add_argument("--max-usd", type=float, help="per step: send nothing that would take it past this")
    cfg, args = cli(__doc__, add_args)
    g, editors = cfg["generate"], {e["name"]: e for e in cfg["edit"]["editors"]}
    sents = pd.read_parquet(path(cfg, "sentences.parquet"))
    sents = sents[sents.split.isin(g["splits"])]
    if args.segment is not None:
        return segment(cfg, sents, args.segment or g["generators"])
    docs = sents.groupby("doc_id").agg(source=("source", "first"), n=("sentence_id", "size")).reset_index()
    if args.pilot:
        docs = docs[docs.doc_id.isin(pilot_docs(cfg, docs, args.pilot))]
    if args.topics:
        text = documents(sents[sents.doc_id.isin(docs.doc_id)])
        prompts = {int(d): TOPIC.format(document=text[d], description=DESCRIPTION[s]) for d, s in
                   zip(docs.doc_id, docs.source)}
        run_step(cfg, editors[g["topic_writer"]], prompts, path(cfg, "generated", "topics.jsonl"), True, args.max_usd)
    topics = accepted(path(cfg, "generated", "topics.jsonl")).set_index("doc_id").text
    for gen in args.generators:
        have = docs[docs.doc_id.isin(topics.index)]
        print(f"{gen}: {len(docs) - len(have)} documents have no topic yet and are skipped")
        prompts = {int(d): ASK.format(genre=GENRE[s], topic=topics[d], n=n) for d, s, n in
                   zip(have.doc_id, have.source, have.n)}
        run_step(cfg, editors[gen], prompts, path(cfg, "generated", f"{gen}.jsonl"), False, args.max_usd)


if __name__ == "__main__":
    main()
