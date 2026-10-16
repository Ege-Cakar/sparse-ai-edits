"""Masked-infilling editor: replaces exactly k words of a sentence with a local masked LM.

The control arm for edit strength. API editors are asked to "change exactly k words" and miss; here k is
exact by construction, so k/L is known without a diff and the k/sqrt(L) vs k/L scaling can be tested directly.

Per sentence (alone by default, like the API editors; with `context: true` the previous and next eligible
sentence are added as context):
  1. Plausibility pass: mask each editable word once; q_i = MLM probability of the best substitute that
     differs from the original, p_i = probability of the original (its first BPE piece, an upper bound for
     multi-piece words). Shared by every (target_r, rep) job of the sentence.
  2. Positions: draw k editable words without replacement, with weight w_i ** position_power, where by
     `positions`:
       uniform       w_i = 1. Forces changes where no good substitute exists ("Police were" -> "Police was"):
                     grammar errors, and edits LOWER LM likelihood, the opposite of AI polishing.
       plausible     w_i = q_i. Fluent, but near-neutral synonym swaps; LM likelihood barely moves.
       mode_seeking  w_i = q_i / p_i (default). Edits where the MLM most prefers another word, i.e. where the
                     human made an unusual choice, and replaces it with the model's preferred word. This is the
                     pull toward the model's mode that AI polishing exerts on the long tail of human writing.
  3. Fill the chosen positions in random order, one at a time on the current text, each with the
     top-1 substitute, so every fill sees the earlier ones.

Editable word: an alphabetic match of text.WORD that, with `protect` (default), is a single MLM vocabulary
entry and is not capitalized mid-sentence. This keeps names and rare content words, which mode_seeking would
otherwise target first ("Loglines" -> "They"), out of reach: AI polishing keeps meaning. Words inside URLs
or e-mail addresses are never edited. A substitute may not repeat a 4+ letter word of the sentence, which
blocks the MLM's repetition failure ("squid-spawning squid"). Substitutes are single alphabetic words, so the
word count is unchanged and text.word_diff counts exactly the k replaced words. When a sentence has fewer than k
editable words, all of them are replaced and word_edits < k_requested (target_r near 1).
Outputs use the edit.py JSONL schema (attempts = 1), plus `positions`, the replaced word indices.
"""
import json

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer

import re

from .text import WORD

URLISH = re.compile(r"\S*(?:://|www\.|@|\.(?:com|org|net|co|uk|html?)\b)\S*", re.I)

def device():
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def word_spans(text, keep=None):
    """(spans of all text.WORD matches, indices of editable ones: alphabetic and keep(word, index))."""
    spans = [m.span() for m in WORD.finditer(text)]
    urls = [m.span() for m in URLISH.finditer(text)]
    return spans, [i for i, (a, b) in enumerate(spans)
                   if text[a:b].isalpha() and not any(u <= a < v for u, v in urls)
                   and (keep is None or keep(text[a:b], i))]


def match_case(new, old):
    if len(old) > 1 and old.isupper():
        return new.upper()
    return new.capitalize() if old[0].isupper() else new


class Infiller:
    def __init__(self, model, batch_size=64, top=50):
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForMaskedLM.from_pretrained(model).to(device()).eval()
        self.batch_size, self.top = batch_size, top
        vocab = self.tok.convert_ids_to_tokens(range(len(self.tok)))
        # Word-initial, purely alphabetic vocabulary entries (BPE marks a leading space with "Ġ").
        ids = [i for i, t in enumerate(vocab) if t.startswith("Ġ") and t[1:].isalpha()]
        self.word_ids = torch.tensor(ids, device=device())
        self.word_str = [vocab[i][1:] for i in ids]
        self.single = set(self.word_str)

    def protected(self, word, i):
        """Names and rare words: capitalized after the first word, or more than one vocabulary piece."""
        return (i > 0 and word[0].isupper()) or word not in self.single

    @torch.no_grad()
    def best_substitutes(self, rows, bans=None):
        """rows: (left, original word, right); bans: per row, lowercase words the substitute may not be.
        Returns [(substitute, q, p_orig)], substitute != original."""
        bans = bans or [set()] * len(rows)
        out = []
        for i in range(0, len(rows), self.batch_size):
            chunk, chunk_bans = rows[i:i + self.batch_size], bans[i:i + self.batch_size]
            texts = [f"{l}{self.tok.mask_token}{r}" for l, _, r in chunk]
            enc = self.tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device())
            logits = self.model(**enc).logits
            b, pos = (enc.input_ids == self.tok.mask_token_id).nonzero(as_tuple=True)
            full = torch.zeros(len(chunk), logits.shape[-1], device=device())
            full[b] = logits[b, pos].float().softmax(-1)
            orig_ids = torch.tensor([self.tok(" " + o, add_special_tokens=False).input_ids[0] for _, o, _ in chunk],
                                    device=device())
            p_orig = full.gather(1, orig_ids[:, None])[:, 0].tolist()
            p, j = full[:, self.word_ids].topk(self.top, dim=-1)
            for (_, orig, _), ban, pr, ix, po in zip(chunk, chunk_bans, p.tolist(), j.tolist(), p_orig):
                seen = ban | {orig.lower()}
                sub, q = next(((self.word_str[k], q) for q, k in zip(pr, ix)
                               if q > 0 and self.word_str[k].lower() not in seen), (None, 0.0))
                out.append((sub, q, po))
        return out


def content_words(text):
    """Lowercase words of 4+ letters: a substitute repeating one of these is the MLM's repetition failure."""
    return {w for w in WORD.findall(text.lower()) if len(w) >= 4}


def masked(context_left, text, span, context_right):
    a, b = span
    left = f"{context_left} {text[:a]}".lstrip()
    return left, text[a:b], f"{text[b:]} {context_right}".rstrip()


def run(infiller, jobs, sents, cfg_editor, seed, out_file):
    """jobs: [(row, r, rep)] from edit.todo_for. Appends one JSON line per job to out_file."""
    from .edit import n_target, rejection
    power = cfg_editor.get("position_power", 1.0)
    strategy = cfg_editor.get("positions", "mode_seeking")
    by_sentence = {}
    for row, r, rep in jobs:
        by_sentence.setdefault(row.sentence_id, []).append((row, r, rep))
    ctx = sents.set_index(["doc_id", "sent_idx"]).text
    def neighbors(row):
        return (ctx.get((row.doc_id, row.sent_idx - 1), "") if cfg_editor.get("context", False) else "",
                ctx.get((row.doc_id, row.sent_idx + 1), "") if cfg_editor.get("context", False) else "")

    for sid, sjobs in tqdm(by_sentence.items(), desc=cfg_editor["name"]):
        row = sjobs[0][0]
        prev, nxt = neighbors(row)
        keep = (lambda w, i: not infiller.protected(w, i)) if cfg_editor.get("protect", True) else None
        spans, editable = word_spans(row.text, keep)
        subs = infiller.best_substitutes([masked(prev, row.text, spans[i], nxt) for i in editable],
                                         [content_words(row.text)] * len(editable))
        q = np.array([x[1] for x in subs])
        p_orig = np.array([x[2] for x in subs])
        weight = {"uniform": np.where(q > 0, 1.0, 0.0), "plausible": q,
                  "mode_seeking": q / np.maximum(p_orig, 1e-12)}[strategy]
        states = []
        for row, r, rep in sjobs:
            rng = np.random.default_rng([seed, int(sid), int(round(r * 1000)), rep])
            k = n_target(r, row.n_words)
            w = weight ** power
            cand = [i for i, x in zip(editable, w) if x > 0]
            w = w[w > 0]
            chosen = rng.choice(cand, size=min(k, len(cand)), replace=False, p=w / w.sum()) if cand else []
            states.append(dict(row=row, r=r, rep=rep, k=k, text=row.text, todo=list(chosen), done=[]))
        # Fill in lockstep across this sentence's jobs: one masked row per unfinished job per step.
        while active := [s for s in states if s["todo"]]:
            rows = []
            for s in active:
                spans_now, _ = word_spans(s["text"])
                rows.append(masked(prev, s["text"], spans_now[s["todo"][0]], nxt))
            bans = [content_words(s["text"]) for s in active]
            for s, (sub, _, _) in zip(active, infiller.best_substitutes(rows, bans)):
                i = s["todo"].pop(0)
                if sub is not None:
                    a, b = word_spans(s["text"])[0][i]
                    s["text"] = s["text"][:a] + match_case(sub, s["text"][a:b]) + s["text"][b:]
                    s["done"].append(int(i))
        for s in states:
            out_file.write(json.dumps(dict(
                sentence_id=int(sid), editor=cfg_editor["name"], target_r=s["r"], rep=s["rep"],
                k_requested=s["k"], text=s["text"], attempts=1, rejected=rejection(s["row"].text, s["text"]),
                positions=sorted(s["done"]))) + "\n")
        out_file.flush()
