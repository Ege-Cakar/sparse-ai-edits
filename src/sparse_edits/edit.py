"""Generate AI edits of sentences at several target strengths through OpenAI-compatible chat endpoints.

Default mode appends raw outputs to edits/<editor>.jsonl and skips finished (sentence_id, target_r, rep)
keys, so it can be rerun after an interruption. Run editors in parallel as separate processes via --editors.
--consolidate measures the actual word changes and writes edits.parquet (every output, with rejection
reason and diff mask) and versions.parquet (human sentences + accepted edits), the input to all detectors.
"""
import asyncio
import json
import os
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
VERSION_COLS = ["sentence_id", "doc_id", "split", "sent_idx", "editor", "target_r", "rep",
                "text", "n_words", "word_edits", "actual_r"]


def n_target(r, n_words):
    return max(1, round(r * n_words))


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


async def edit_one(client, editor, row, r, rep, max_attempts, sem):
    """One edit; resamples up to max_attempts on rejection. Returns None on API failure (retried next run)."""
    k = n_target(r, row.n_words)
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": PROMPT.format(k=k, n=row.n_words, text=row.text)}]
    async with sem:
        for attempt in range(1, max_attempts + 1):
            try:
                resp = await client.chat.completions.create(
                    model=editor["model"], messages=messages, **editor.get("params", {}))
            except openai.APIError as e:
                print(f"{editor['name']} sentence {row.sentence_id}: {e!r}")
                return None
            out = clean(resp.choices[0].message.content, row.text)
            if (reason := rejection(row.text, out)) is None:
                break
    return dict(sentence_id=int(row.sentence_id), editor=editor["name"], target_r=r, rep=rep,
                k_requested=k, text=out, attempts=attempt, rejected=reason)


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


async def run_editor(cfg, editor, todo, out_path):
    e = cfg["edit"]
    client = openai.AsyncOpenAI(base_url=os.environ[editor["base_url_env"]] if "base_url_env" in editor else None,
                                api_key=os.environ[editor["api_key_env"]], max_retries=8)
    sem, failed = asyncio.Semaphore(e["concurrency"]), 0
    if out_path.exists() and not out_path.read_text().endswith("\n"):
        out_path.write_text(out_path.read_text() + "\n")  # terminate a truncated last line
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


def consolidate(cfg):
    sents = pd.read_parquet(path(cfg, "sentences.parquet"))
    edits = pd.DataFrame([r for p in sorted(Path(cfg["data_dir"], "edits").glob("*.jsonl")) for r in read_jsonl(p)])
    edits = edits.drop_duplicates(["sentence_id", "editor", "target_r", "rep"], keep="last")
    orig = sents.set_index("sentence_id")
    edits = edits.join(orig[["doc_id", "split", "sent_idx"]], on="sentence_id")
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
        todo = todo_for(cfg, sents, editor, out_path)
        print(f"{editor['name']}: {len(todo)} requests to send")
        if not args.dry_run:
            asyncio.run(run_editor(cfg, editor, todo, out_path))


if __name__ == "__main__":
    main()
