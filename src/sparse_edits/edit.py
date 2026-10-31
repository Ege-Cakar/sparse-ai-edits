"""Generate AI edits of sentences through OpenAI-compatible chat endpoints.

The prompt asks for an edit without a requested size (no "change k of n words"), so each editor edits at its own
strength, measured afterwards (actual_r in --consolidate); target_r is empty for these edits. A config may still set
target_strengths (the masked-LM control in edit_mlm.py uses them).

Each request shows the sentence inside its whole (original, human) document, with the document first so that every
request for one document shares a cacheable prefix; jobs are sent in document order for the same reason.

Default mode sends concurrent requests; --batch uses the Batch API instead (half price at OpenAI, results within
24 h), resubmitting rejected outputs in further batches. Both append outputs to edits/<editor>.jsonl and skip finished
(sentence_id, target_r, rep) keys, so they can be rerun after an interruption; --batch also records submitted batch
ids in edits/<editor>.jsonl.batches and collects them on the next run. Run editors in parallel via --editors.
--consolidate measures the actual word changes and writes edits.parquet (every output, with rejection
reason and diff mask) and versions.parquet (human sentences + accepted edits), the input to all detectors.
"""
import asyncio
import json
import os
import re
import time
from pathlib import Path

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
# The document comes first, so every request for one document shares a cacheable prefix.
PROMPT = "Text: {document}\n\n{ask} Reply with only the edited sentence, followed by a line break."
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
    """Chat-completions body asking the editor to edit row.text, shown inside its whole document; essay placeholders
    get an instruction to keep them."""
    prompt = PROMPT.format(document=document, ask=ASK[editor.get("prompt", "edit")].format(sentence=row.text))
    if keep := placeholders(row):
        prompt += f" Keep the placeholder{'s' * (len(keep) > 1)} {', '.join(keep)} exactly as written."
    return dict(model=editor["model"], messages=[{"role": "system", "content": SYSTEM.format(product=editor["product"])},
                                                 {"role": "user", "content": prompt}], **editor.get("params", {}))


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


def submit(client, editor, by_id, docs, jobs):
    """Upload jobs (sentence_id, target_r, rep, attempt) as one batch; returns the batch id."""
    lines = [json.dumps({"custom_id": f"{sid}:{r}:{rep}:{attempt}", "method": "POST", "url": "/v1/chat/completions",
                         "body": request(editor, by_id.loc[sid], docs[by_id.at[sid, "doc_id"]])})
             for sid, r, rep, attempt in jobs]
    file = client.files.create(file=("requests.jsonl", "\n".join(lines).encode()), purpose="batch")
    batch = client.batches.create(input_file_id=file.id, endpoint="/v1/chat/completions", completion_window="24h")
    print(f"{editor['name']}: submitted {batch.id} with {len(jobs)} requests")
    return batch.id


def collect(client, editor, by_id, batch_id, out_path, max_attempts):
    """Wait for a batch, append accepted and out-of-attempts outputs to out_path, return the jobs to resample.
    Failed requests are not written, so the next run retries them."""
    while (batch := client.batches.retrieve(batch_id)).status not in ("completed", "failed", "expired", "cancelled"):
        time.sleep(60)
    print(f"{editor['name']}: {batch_id} {batch.status}, {batch.request_counts}")
    retry, lines = [], client.files.content(batch.output_file_id).text.splitlines() if batch.output_file_id else []
    terminate_last_line(out_path)
    with open(out_path, "a") as f:
        for res in map(json.loads, lines):
            if (res["response"] or {}).get("status_code") != 200:
                continue
            sid, r, rep, attempt = (t(v) for t, v in zip((int, strength, int, int), res["custom_id"].split(":")))
            row = by_id.loc[sid]
            out = clean(res["response"]["body"]["choices"][0]["message"]["content"], row.text)
            if (reason := rejection(row.text, out, placeholders(row))) and attempt < max_attempts:
                retry.append((sid, r, rep, attempt + 1))
            else:
                f.write(json.dumps(record(editor, row, r, rep, out, attempt, reason)) + "\n")
    return retry


def strength(v):
    return None if v == "None" else float(v)


def run_batches(cfg, editor, by_id, docs, jobs, out_path):
    """Submit jobs in batches of edit.batch_size, collect them, and repeat with the jobs to resample. Batches already
    recorded in the state file are collected first."""
    client, state, size = openai.OpenAI(**client_args(editor)), Path(f"{out_path}.batches"), cfg["edit"]["batch_size"]
    while jobs or state.exists():
        if not state.exists():
            state.write_text(json.dumps([submit(client, editor, by_id, docs, jobs[i:i + size])
                                         for i in range(0, len(jobs), size)]))
        jobs = [job for batch_id in json.loads(state.read_text())
                for job in collect(client, editor, by_id, batch_id, out_path, cfg["edit"]["max_attempts"])]
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
        out_path = path(cfg, "edits", f"{editor['name']}.jsonl")
        if args.batch and not args.dry_run:
            run_batches(cfg, editor, by_id, docs, [], out_path)  # pending batches
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
            run_batches(cfg, editor, by_id, docs, [(row.sentence_id, r, rep, 1) for row, r, rep in todo], out_path)
        else:
            asyncio.run(run_editor(cfg, editor, docs, todo, out_path))


if __name__ == "__main__":
    main()
