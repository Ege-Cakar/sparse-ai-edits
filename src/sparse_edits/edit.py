"""Generate AI edits of sentences at several target strengths through OpenAI-compatible chat endpoints.

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
import time
from pathlib import Path

import openai
import pandas as pd
from tqdm import tqdm

from . import cli, path
from .text import sentences, word_diff, words

SYSTEM = "You are an AI writing assistant that edits people's writing."
PROMPT = (
    "Edit the sentence below as you would when polishing someone's writing. "
    "Change exactly {k} of its {n} words by replacing, inserting, or deleting words. "
    "Keep the meaning and keep it one sentence. Reply with only the edited sentence.\n\n"
    "Sentence: {text}"
)
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


def rejection(original, edited):
    if not edited:
        return "empty"
    if len(sentences(edited)) != 1:
        return "not_one_sentence"
    if words(edited) == words(original):
        return "unchanged"
    return None


def client_args(editor):
    key = os.environ[editor["api_key_env"]]
    base_url = editor.get("base_url") or (os.environ[editor["base_url_env"]] if "base_url_env" in editor else None)
    headers = {editor["auth_header"]: key} if "auth_header" in editor else None   # gateways such as HUIT's
    return dict(base_url=base_url, api_key=key, default_headers=headers, max_retries=8)


def request(editor, row, r):
    """Chat-completions body asking the editor to change k = n_target(r, n_words) words of row.text."""
    prompt = PROMPT.format(k=n_target(r, row.n_words), n=row.n_words, text=row.text)
    return dict(model=editor["model"], messages=[{"role": "system", "content": SYSTEM},
                                                 {"role": "user", "content": prompt}], **editor.get("params", {}))


def record(editor, row, r, rep, out, attempt, reason):
    return dict(sentence_id=int(row.sentence_id), editor=editor["name"], target_r=r, rep=rep,
                k_requested=n_target(r, row.n_words), text=out, attempts=attempt, rejected=reason)


async def edit_one(client, editor, row, r, rep, max_attempts, sem):
    """One edit; resamples up to max_attempts on rejection. Returns None on API failure (retried next run)."""
    async with sem:
        for attempt in range(1, max_attempts + 1):
            try:
                resp = await client.chat.completions.create(**request(editor, row, r))
            except openai.APIError as e:
                print(f"{editor['name']} sentence {row.sentence_id}: {e!r}")
                return None
            if isinstance(resp, str):   # the HUIT gateway occasionally returns a non-JSON body with status 200
                print(f"{editor['name']} sentence {row.sentence_id}: non-JSON response {resp[:300]!r}")
                return None
            out = clean(resp.choices[0].message.content, row.text)
            if (reason := rejection(row.text, out)) is None:
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
    e = cfg["edit"]
    done = {(d["sentence_id"], d["target_r"], d["rep"]) for d in read_jsonl(out_path)}
    return [(row, r, rep) for row in sents[sents.split.isin(e["splits"])].itertuples()
            for r in e["target_strengths"] for rep in range(e["reps"])
            if (row.sentence_id, r, rep) not in done]


def terminate_last_line(out_path):
    if out_path.exists() and not out_path.read_text().endswith("\n"):
        out_path.write_text(out_path.read_text() + "\n")  # a line truncated by an interrupted run


async def run_editor(cfg, editor, todo, out_path):
    e = cfg["edit"]
    client = openai.AsyncOpenAI(**client_args(editor))
    sem, failed = asyncio.Semaphore(e["concurrency"]), 0
    terminate_last_line(out_path)
    with open(out_path, "a") as f, tqdm(total=len(todo), desc=editor["name"]) as bar:
        for i in range(0, len(todo), e["chunk_size"]):
            jobs = [edit_one(client, editor, *job, e["max_attempts"], sem) for job in todo[i:i + e["chunk_size"]]]
            for fut in asyncio.as_completed(jobs):
                res = await fut
                failed += res is None
                if res is not None:
                    f.write(json.dumps(res) + "\n")
                bar.update()
            f.flush()
    print(f"{editor['name']}: {failed} requests failed after retries; rerun to retry them")


def submit(client, editor, by_id, jobs):
    """Upload jobs (sentence_id, target_r, rep, attempt) as one batch; returns the batch id."""
    lines = [json.dumps({"custom_id": f"{sid}:{r}:{rep}:{attempt}", "method": "POST", "url": "/v1/chat/completions",
                         "body": request(editor, by_id.loc[sid], r)}) for sid, r, rep, attempt in jobs]
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
            sid, r, rep, attempt = (t(v) for t, v in zip((int, float, int, int), res["custom_id"].split(":")))
            row = by_id.loc[sid]
            out = clean(res["response"]["body"]["choices"][0]["message"]["content"], row.text)
            if (reason := rejection(row.text, out)) and attempt < max_attempts:
                retry.append((sid, r, rep, attempt + 1))
            else:
                f.write(json.dumps(record(editor, row, r, rep, out, attempt, reason)) + "\n")
    return retry


def run_batches(cfg, editor, by_id, jobs, out_path):
    """Submit jobs in batches of edit.batch_size, collect them, and repeat with the jobs to resample. Batches already
    recorded in the state file are collected first."""
    client, state, size = openai.OpenAI(**client_args(editor)), Path(f"{out_path}.batches"), cfg["edit"]["batch_size"]
    while jobs or state.exists():
        if not state.exists():
            state.write_text(json.dumps([submit(client, editor, by_id, jobs[i:i + size])
                                         for i in range(0, len(jobs), size)]))
        jobs = [job for batch_id in json.loads(state.read_text())
                for job in collect(client, editor, by_id, batch_id, out_path, cfg["edit"]["max_attempts"])]
        state.unlink()


def consolidate(cfg):
    sents = pd.read_parquet(path(cfg, "sentences.parquet"))
    edits = pd.DataFrame([r for p in sorted(Path(cfg["data_dir"], "edits").glob("*.jsonl")) for r in read_jsonl(p)])
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

    print(edits.groupby(["editor", "target_r"]).rejected.value_counts(dropna=False).unstack(fill_value=0))
    print(accepted.groupby(["editor", "target_r"]).actual_r.describe()[["count", "mean", "25%", "50%", "75%"]])


def main():
    def add_args(p):
        p.add_argument("--editors", nargs="*", help="editor names to run (default: all in config)")
        p.add_argument("--dry-run", action="store_true", help="print request counts and exit")
        p.add_argument("--batch", action="store_true", help="use the Batch API instead of concurrent requests")
        p.add_argument("--limit", type=int, help="send at most this many requests, whole sentences in seeded random order")
        p.add_argument("--consolidate", action="store_true")
    cfg, args = cli(__doc__, add_args)
    if args.consolidate:
        return consolidate(cfg)
    if not cfg["edit"]["editors"]:
        raise SystemExit("edit.editors is empty; add editors to the config (see docs/review.md)")
    sents = pd.read_parquet(path(cfg, "sentences.parquet"))
    for editor in cfg["edit"]["editors"]:
        if args.editors and editor["name"] not in args.editors:
            continue
        out_path = path(cfg, "edits", f"{editor['name']}.jsonl")
        if args.batch and not args.dry_run:
            run_batches(cfg, editor, sents.set_index("sentence_id", drop=False), [], out_path)  # pending batches
        todo = todo_for(cfg, sents, editor, out_path)
        if args.limit is not None:   # sentences in a seeded random order, so a budget cap does not cut whole sources
            order = sents.sentence_id.sample(frac=1, random_state=cfg["seed"]).reset_index(drop=True)
            rank = dict(zip(order, order.index))
            todo = sorted(todo, key=lambda job: rank[job[0].sentence_id])[:args.limit]
        print(f"{editor['name']}: {len(todo)} requests to send")
        if args.dry_run:
            continue
        if args.batch:
            run_batches(cfg, editor, sents.set_index("sentence_id", drop=False),
                        [(row.sentence_id, r, rep, 1) for row, r, rep in todo], out_path)
        else:
            asyncio.run(run_editor(cfg, editor, todo, out_path))


if __name__ == "__main__":
    main()
