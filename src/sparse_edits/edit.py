"""Generate AI edits of sentences through OpenAI-compatible chat endpoints, or Anthropic's Messages API (api: anthropic).

The prompt asks for an edit without a requested size (no "change k of n words"), so each editor edits at its own
strength, measured afterwards (actual_r in --consolidate); target_r is empty for these edits. A config may still set
target_strengths (the masked-LM control in edit_mlm.py uses them).

Each request shows the sentence inside its whole (original, human) document, with the document first so that every
request for one document shares a cacheable prefix; jobs are sent in document order for the same reason.

Default mode sends concurrent requests; --batch uses the Batch API instead (half price at OpenAI and Anthropic, results
within 24 h), resubmitting rejected outputs in further batches; Anthropic editors run only with --batch. Both append
outputs to edits/<editor>.jsonl and skip finished (sentence_id, target_r, rep) keys, so they can be rerun after an
interruption; --batch also records each batch id in edits/<editor>.jsonl.batches as it is submitted and collects
those batches on the next run, so no request is submitted twice, and logs each response's token usage (cache reads
and writes included) to edits/<editor>.jsonl.usage. Run editors in parallel via --editors.
--consolidate measures the actual word changes and writes edits.parquet (every output, with rejection
reason and diff mask) and versions.parquet (human sentences + accepted edits), the input to all detectors.
"""
import asyncio
import json
import os
import re
import time
from itertools import islice
from pathlib import Path

import anthropic
import openai
import pandas as pd
from tqdm import tqdm

from . import cli, path
from .sources import SOURCES
from .text import sentences, word_diff, words

SYSTEM = """You are {product}, a helpful AI assistant. Answer the user's questions clearly, accurately, and concisely.

Guidelines:
- If you don't know something or aren't sure, say so rather than guessing.
- If a request is vague, reply as accurately as you can with your best reading of it.
- Match the length of your answer to the complexity of the question.
- Use plain language; use formatting (lists, code blocks) only when it helps readability.
- Decline requests that are harmful or illegal, and briefly explain why.

Today's date is November 1, 2026."""
ASK = {   # the instruction per editor `prompt` (default "edit"); wording under review (docs/review.md item 15)
    "edit": 'Edit the sentence "{sentence}", please in the text above, to improve the writing. '
            "Keep the meaning and about the same length, and keep it one sentence.",
    "light": 'Lightly edit the sentence "{sentence}", please in the text above, to improve the writing, changing only '
             "what needs it. Keep the meaning and about the same length, and keep it one sentence.",
}
# The user message, as (document part, instruction part): the document comes first, so every request for one document
# shares a cacheable prefix (marked for caching in Anthropic requests).
PROMPT = "Text: {document}", "\n\n{ask} Reply with only the edited sentence, followed by a line break."
PLACEHOLDER = re.compile(r"\bGeneric_\w+|\b[A-Z]+(?:_[A-Z]+)+\b")   # anonymized names etc. in the essay datasets
PLACEHOLDER_SOURCES = {"persuade", "ellipse"}
VERSION_COLS = ["sentence_id", "source", "doc_id", "split", "sent_idx", "editor", "target_r", "rep",
                "text", "n_words", "word_edits", "actual_r"]


def n_target(r, n_words):
    return max(1, round(r * int(n_words)))


def clean(out, original):
    """Strip whitespace and one pair of wrapping quotes that the original sentence did not have."""
    out = (out or "").strip()
    if len(out) > 1 and out[0] in '"“' and out[-1] in '"”' and original[0] not in '"“':
        out = out[1:-1].strip()
    return out


def placeholders(row):
    """Anonymization placeholders in an essay sentence (e.g. Generic_Name), which the editor is asked to keep."""
    return sorted(set(PLACEHOLDER.findall(row.text))) if row.source in PLACEHOLDER_SOURCES else []


def rejection(original, edited, keep=()):
    if not edited:
        return "empty"
    if len(sentences(edited)) != len(sentences(original)):   # corpus sentences are SaT; Punkt may read one as two
        return "sentence_count_changed"
    if words(edited) == words(original):
        return "unchanged"
    if len(words(edited)) > 2 * len(words(original)):   # e.g. "Thanks." -> a new sentence of invented content
        return "too_long"
    if any(p not in edited for p in keep):
        return "placeholder_changed"
    return None


def documents(sents):
    """Each document's text as the editor sees it: a paragraph's sentences joined by spaces, paragraphs joined by the
    source's paragraph separator."""
    paras = sents.groupby(["doc_id", "para_idx"]).agg(source=("source", "first"), text=("text", " ".join))
    return {doc: SOURCES[g.source.iat[0]][1].join(g.text) for doc, g in paras.groupby(level="doc_id")}


def client_args(editor):
    key = os.environ[editor["api_key_env"]] if "api_key_env" in editor else "EMPTY"   # self-hosted vLLM needs none
    base_url = editor.get("base_url") or (os.environ[editor["base_url_env"]] if "base_url_env" in editor else None)
    headers = {editor["auth_header"]: key} if "auth_header" in editor else None   # gateways such as HUIT's
    return dict(base_url=base_url, api_key=key, default_headers=headers, max_retries=8)


def request(editor, row, document):
    """Request body asking the editor to edit row.text, shown inside its whole document; essay placeholders get an
    instruction to keep them. Chat completions by default; for Anthropic, the same text with the document part
    (after the system prompt) cached. cache_key: true sends a per-document prompt_cache_key (OpenAI)."""
    head = PROMPT[0].format(document=document)
    ask = PROMPT[1].format(ask=ASK[editor.get("prompt", "edit")].format(sentence=row.text))
    if keep := placeholders(row):
        ask += f" Keep the placeholder{'s' * (len(keep) > 1)} {', '.join(keep)} exactly as written."
    system = SYSTEM.format(product=editor["product"])
    if editor.get("api") == "anthropic":
        user = [{"type": "text", "text": head, "cache_control": {"type": "ephemeral"}}, {"type": "text", "text": ask}]
        return dict(model=editor["model"], system=system, messages=[{"role": "user", "content": user}],
                    **editor.get("params", {}))
    body = dict(model=editor["model"], messages=[{"role": "system", "content": system},
                                                 {"role": "user", "content": head + ask}], **editor.get("params", {}))
    return body | ({"prompt_cache_key": f"doc-{row.doc_id}"} if editor.get("cache_key") else {})


def record(editor, row, r, rep, out, attempt, reason):
    return dict(sentence_id=int(row.sentence_id), editor=editor["name"], target_r=r, rep=rep, text=out,
                attempts=attempt, rejected=reason)


async def edit_one(client, editor, docs, row, r, rep, max_attempts, sem):
    """One edit; resamples up to max_attempts on rejection. Returns None on API failure (retried next run)."""
    async with sem:
        for attempt in range(1, max_attempts + 1):
            try:
                resp = await client.chat.completions.create(**request(editor, row, docs[row.doc_id]))
            except openai.APIError as e:
                print(f"{editor['name']} sentence {row.sentence_id}: {e!r}")
                return None
            if isinstance(resp, str):   # the HUIT gateway occasionally returns a non-JSON body with status 200
                print(f"{editor['name']} sentence {row.sentence_id}: non-JSON response {resp[:300]!r}")
                return None
            out = clean(resp.choices[0].message.content, row.text)
            if (reason := rejection(row.text, out, placeholders(row))) is None:
                break
    return record(editor, row, r, rep, out, attempt, reason)


def read_jsonl(p):
    """Rows of a JSONL file, skipping a line truncated by an interrupted run."""
    rows = []
    for line in (p.read_text().splitlines() if p.exists() else []):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def todo_for(cfg, sents, editor, out_path):
    """(row, target_r, rep) jobs not yet in out_path; target_r is None unless target_strengths are configured.
    Sentences with no words ("!", "***") are never sent; they stay human in every version."""
    e = cfg["edit"]
    done = {(d["sentence_id"], d["target_r"], d["rep"]) for d in read_jsonl(out_path)}
    return [(row, r, rep) for row in sents[sents.split.isin(e["splits"]) & (sents.n_words > 0)].itertuples()
            for r in editor.get("target_strengths", e.get("target_strengths", [None])) for rep in range(e["reps"])
            if (row.sentence_id, r, rep) not in done]


def terminate_last_line(out_path):
    if out_path.exists() and not out_path.read_text().endswith("\n"):
        out_path.write_text(out_path.read_text() + "\n")  # a line truncated by an interrupted run


async def run_editor(cfg, editor, docs, todo, out_path):
    e = cfg["edit"]
    client = openai.AsyncOpenAI(**client_args(editor))
    sem, failed = asyncio.Semaphore(e["concurrency"]), 0
    terminate_last_line(out_path)
    with open(out_path, "a", buffering=1) as f, tqdm(total=len(todo), desc=editor["name"]) as bar:   # a line per record
        for i in range(0, len(todo), e["chunk_size"]):
            jobs = [edit_one(client, editor, docs, *job, e["max_attempts"], sem) for job in todo[i:i + e["chunk_size"]]]
            for fut in asyncio.as_completed(jobs):
                res = await fut
                failed += res is None
                if res is not None:
                    f.write(json.dumps(res) + "\n")
                bar.update()
            f.flush()
    print(f"{editor['name']}: {failed} requests failed after retries; rerun to retry them")


def custom_id(sid, r, rep, attempt):
    """Batch request id; Anthropic allows only letters, digits, _ and -."""
    return f"{sid}_{r}_{rep}_{attempt}".replace(".", "p")


def parse_id(cid):
    sid, r, rep, attempt = cid.replace("p", ".").split("_")
    return int(sid), None if r == "None" else float(r), int(rep), int(attempt)


def groups(editor, by_id, docs, jobs, size, max_bytes):
    """(custom_id, body) pairs for jobs (sentence_id, target_r, rep, attempt), in groups under a batch's request count
    and size limits (each request carries its whole document)."""
    group, n = [], 0
    for job in jobs:
        body = request(editor, by_id.loc[job[0]], docs[by_id.at[job[0], "doc_id"]])
        b = len(json.dumps(body)) + 200
        if group and (len(group) == size or n + b > max_bytes):
            yield group
            group, n = [], 0
        group.append((custom_id(*job), body))
        n += b
    if group:
        yield group


def submit(client, editor, group):
    """Submit one batch; returns its id."""
    if editor.get("api") == "anthropic":
        batch_id = client.messages.batches.create(requests=[{"custom_id": c, "params": b} for c, b in group]).id
    else:
        lines = [json.dumps({"custom_id": c, "method": "POST", "url": "/v1/chat/completions", "body": b})
                 for c, b in group]
        file = client.files.create(file=("requests.jsonl", "\n".join(lines).encode()), purpose="batch")
        batch_id = client.batches.create(input_file_id=file.id, endpoint="/v1/chat/completions",
                                         completion_window="24h").id
    print(f"{editor['name']}: submitted {batch_id} with {len(group)} requests")
    return batch_id


def results(client, editor, batch_id):
    """Wait for a batch; yield (custom_id, reply text, token usage) per succeeded request."""
    if editor.get("api") == "anthropic":
        while (batch := client.messages.batches.retrieve(batch_id)).processing_status != "ended":
            time.sleep(60)
        print(f"{editor['name']}: {batch_id} ended, {batch.request_counts}")
        for res in client.messages.batches.results(batch_id):
            if res.result.type == "succeeded":
                m = res.result.message
                yield res.custom_id, "".join(b.text for b in m.content if b.type == "text"), m.usage.model_dump()
        return
    while (batch := client.batches.retrieve(batch_id)).status not in ("completed", "failed", "expired", "cancelled"):
        time.sleep(60)
    print(f"{editor['name']}: {batch_id} {batch.status}, {batch.request_counts}")
    for res in map(json.loads, client.files.content(batch.output_file_id).text.splitlines() if batch.output_file_id
                   else []):
        if (res["response"] or {}).get("status_code") == 200:
            body = res["response"]["body"]
            yield res["custom_id"], body["choices"][0]["message"]["content"], body["usage"]


def collect(client, editor, by_id, batch_id, out_path, max_attempts):
    """Wait for a batch, append accepted and out-of-attempts outputs to out_path and each response's usage to
    out_path.usage, return the jobs to resample. Failed requests are not written, so the next run retries them."""
    retry = []
    terminate_last_line(out_path)
    with open(out_path, "a", buffering=1) as f, open(f"{out_path}.usage", "a") as u:
        for cid, text, usage in results(client, editor, batch_id):
            u.write(json.dumps(dict(custom_id=cid, batch_id=batch_id, usage=usage)) + "\n")
            sid, r, rep, attempt = parse_id(cid)
            row = by_id.loc[sid]
            out = clean(text, row.text)
            if (reason := rejection(row.text, out, placeholders(row))) and attempt < max_attempts:
                retry.append((sid, r, rep, attempt + 1))
            else:
                f.write(json.dumps(record(editor, row, r, rep, out, attempt, reason)) + "\n")
    return retry


def spent(editor, out_path):
    """(dollars, responses) so far, from out_path.usage and the editor's batch prices per million tokens
    (price: input, cache_read, cache_write, output)."""
    p, cost, n = editor["price"], 0.0, 0
    for line in read_jsonl(Path(f"{out_path}.usage")):
        u, n = line["usage"], n + 1
        if editor.get("api") == "anthropic":
            tokens = dict(input=u["input_tokens"], cache_read=u["cache_read_input_tokens"] or 0,
                          cache_write=u["cache_creation_input_tokens"] or 0, output=u["output_tokens"])
        else:
            cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
            tokens = dict(input=u["prompt_tokens"] - cached, cache_read=cached, output=u["completion_tokens"])
        cost += sum(p.get(k, 0) * v for k, v in tokens.items()) / 1e6
    return cost, n


def run_batches(cfg, editor, by_id, docs, jobs, out_path, max_usd=None):
    """Submit jobs in batches under edit.batch_size requests and edit.batch_mb MB (the editor's batch_limits override
    these), at most max_queued batches per round, recording each batch id in the state file as soon as it is
    submitted; collect them and repeat with the jobs not yet sent and those to resample. Batches already in the state
    file are collected first. With max_usd, a batch is submitted only if the spend so far plus the batch at the cost
    per request measured so far (the pilot's, at first) stays under it."""
    e, state = cfg["edit"] | editor.get("batch_limits", {}), Path(f"{out_path}.batches")
    client = (anthropic.Anthropic(api_key=os.environ[editor["api_key_env"]], max_retries=8)
              if editor.get("api") == "anthropic" else openai.OpenAI(**client_args(editor)))
    while jobs or state.exists():
        cost, n = spent(editor, out_path) if max_usd else (0.0, 0)
        rate = cost / n if n else None   # dollars per request so far, fixed for this round
        sent = set()
        for group in islice(groups(editor, by_id, docs, jobs, e["batch_size"], e["batch_mb"] * 2**20),
                            e.get("max_queued")):
            if max_usd and (rate is None or cost + rate * len(group) > max_usd):
                raise SystemExit(f"{editor['name']}: ${cost:.2f} spent or committed; the next {len(group)} requests "
                                 f"would pass ${max_usd} (or no usage measured yet)")
            cost += rate * len(group) if max_usd else 0
            ids = json.loads(state.read_text()) if state.exists() else []
            state.write_text(json.dumps(ids + [submit(client, editor, group)]))
            sent.update(cid for cid, _ in group)
        jobs = [job for job in jobs if custom_id(*job) not in sent] + [
            job for batch_id in json.loads(state.read_text())
            for job in collect(client, editor, by_id, batch_id, out_path, e["max_attempts"])]
        state.unlink()


def consolidate(cfg):
    sents = pd.read_parquet(path(cfg, "sentences.parquet"))
    edits = pd.DataFrame([r for p in sorted(Path(cfg["data_dir"], "edits").glob("*.jsonl")) for r in read_jsonl(p)])
    edits = edits.assign(target_r=edits.target_r.astype(float))   # NaN: no requested strength
    edits = edits.drop_duplicates(["sentence_id", "editor", "target_r", "rep"], keep="last")
    orig = sents.set_index("sentence_id")
    edits = edits.join(orig[["source", "doc_id", "split", "sent_idx"]], on="sentence_id")
    diffs = [word_diff(o, t) for o, t in zip(orig.text.loc[edits.sentence_id], edits.text)]
    edits["word_edits"], edits["ai_word_mask"] = [d for d, _ in diffs], [m for _, m in diffs]
    edits["n_words"] = edits.text.map(lambda t: len(words(t)))
    edits["actual_r"] = edits.word_edits / orig.n_words.loc[edits.sentence_id].to_numpy()
    edits.to_parquet(path(cfg, "edits.parquet"), index=False)

    accepted = edits[edits.rejected.isna()]
    human = sents.assign(editor="human", target_r=0.0, rep=0, word_edits=0, actual_r=0.0)
    versions = pd.concat([human[VERSION_COLS], accepted[VERSION_COLS]], ignore_index=True)
    versions.insert(0, "version_id", range(len(versions)))
    versions.to_parquet(path(cfg, "versions.parquet"), index=False)

    by = dict(by=["editor", "target_r"], dropna=False)
    print(edits.groupby(**by).rejected.value_counts(dropna=False).unstack(fill_value=0))
    print(accepted.groupby(**by).actual_r.describe()[["count", "mean", "25%", "50%", "75%"]])


def coverage(cfg, sents):
    """Per (editor, source, split): `passes`, the (target_r, rep) passes in which every editable sentence of the cell
    has a record (accepted or out of attempts), and `edits_per_sentence`, records per editable sentence."""
    edits = pd.DataFrame([r for p in sorted(Path(cfg["data_dir"], "edits").glob("*.jsonl")) for r in read_jsonl(p)])
    editable = sents[sents.split.isin(cfg["edit"]["splits"]) & (sents.n_words > 0)]
    n = editable.groupby(["source", "split"]).size().rename("n").reset_index()
    if edits.empty:
        return pd.DataFrame()
    done = edits.drop_duplicates(["editor", "sentence_id", "target_r", "rep"]).merge(
        editable[["sentence_id", "source", "split"]])
    c = done.groupby(["editor", "source", "split", "target_r", "rep"], dropna=False).size().rename("done").reset_index()
    c = c.merge(n)
    t = c.assign(complete=c.done >= c.n).groupby(["editor", "source", "split"]).agg(
        passes=("complete", "sum"), done=("done", "sum"), n=("n", "first"))
    return t.assign(edits_per_sentence=(t.done / t.n).round(3)).drop(columns=["done", "n"])


def main():
    def add_args(p):
        p.add_argument("--editors", nargs="*", help="editor names to run (default: all in config)")
        p.add_argument("--dry-run", action="store_true", help="print request counts and exit")
        p.add_argument("--batch", action="store_true", help="use the Batch API instead of concurrent requests")
        p.add_argument("--max-usd", type=float, help="--batch: submit no batch that would take this editor past it")
        p.add_argument("--limit", type=int, help="send at most this many requests, whole documents in seeded random order")
        p.add_argument("--consolidate", action="store_true")
        p.add_argument("--status", action="store_true", help="print passes per editor, source and split")
    cfg, args = cli(__doc__, add_args)
    if args.consolidate:
        return consolidate(cfg)
    if args.status:
        return print(coverage(cfg, pd.read_parquet(path(cfg, "sentences.parquet"))).to_string())
    if not cfg["edit"]["editors"]:
        raise SystemExit("edit.editors is empty; add editors to the config (see docs/review.md)")
    sents = pd.read_parquet(path(cfg, "sentences.parquet"))
    docs, by_id = documents(sents), sents.set_index("sentence_id", drop=False)
    for editor in cfg["edit"]["editors"]:
        if args.editors and editor["name"] not in args.editors:
            continue
        if editor.get("type") == "mlm":   # the optional local control editor: only when named explicitly
            if editor["name"] not in (args.editors or []):
                print(f"{editor['name']}: masked-LM control editor, runs only when named with --editors")
                continue
            if args.batch:
                raise SystemExit("--batch is for API editors; the masked-LM editor runs locally")
        if editor.get("api") == "anthropic" and not args.batch and not args.dry_run:
            raise SystemExit(f"{editor['name']}: Anthropic editors run only with --batch")
        out_path = path(cfg, "edits", f"{editor['name']}.jsonl")
        if args.batch and not args.dry_run:
            run_batches(cfg, editor, by_id, docs, [], out_path, args.max_usd)  # pending batches, then their retries
        todo = todo_for(cfg, sents, editor, out_path)   # in document order, so a document's requests share its cache
        if args.limit is not None:   # whole documents in a seeded random order, so a budget cap does not cut sources
            order = sents.doc_id.drop_duplicates().sample(frac=1, random_state=cfg["seed"])
            rank = dict(zip(order, range(len(order))))
            todo = sorted(todo, key=lambda job: rank[job[0].doc_id])[:args.limit]   # stable: keeps document order
        print(f"{editor['name']}: {len(todo)} requests to send")
        if args.dry_run:
            continue
        if editor.get("type") == "mlm":   # local masked-infilling editor, no API
            from .edit_mlm import Infiller, run
            with open(out_path, "a") as f:
                run(Infiller(editor["model"], editor.get("batch_size", 64)), todo, sents, editor, cfg["seed"], f)
        elif args.batch:
            run_batches(cfg, editor, by_id, docs, [(row.sentence_id, r, rep, 1) for row, r, rep in todo], out_path,
                        args.max_usd)
        else:
            asyncio.run(run_editor(cfg, editor, docs, todo, out_path))


if __name__ == "__main__":
    main()
