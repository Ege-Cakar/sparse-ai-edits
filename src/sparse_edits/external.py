"""Released AI-text detectors with their own heads (external.detectors), at the sentence and the document level.

--sentences  scores/<name>.parquet (version_id, score) for every version in external.splits, each sentence alone:
             the input of calibrate.py and phase.py, like any detector.
--documents  docscores/<name>.parquet: the calib split's human documents (the null) and the test documents of phase.py's
             whole mode, each with at least phase.whole_min_sentences sentences. Each test document appears human
             (theta inf) and, for every editor and theta, with phase.py's edited positions (the same cell seeds and
             draws; a chosen sentence without an accepted edit stays human), joined with the corpus's separators and
             scored whole.
Scores are saved in chunks to external_parts/, so a preempted run resumes.
--doc-tests  docscores/<name>_tests.parquet: each test document's conformal p-value against the calib documents of its
             source, and rejection at phase.alpha. CPU only.
Detectors (kind), each run as its authors run it:
- editlens: Pangram's EditLens adapter on Llama-3.2-3B, as scripts/inference.py and configs/llama.yaml of
  github.com/pangramlabs/EditLens: the base in 4-bit NF4 with bf16 compute, the bucket head (LayerNorm then linear,
  their NormedLinear, fp32) under bf16 autocast, their clean_text, at most 1024 tokens. Score = expected bucket /
  (buckets - 1).
- desklib: DeBERTa-v3-large, mean pooling, one logit (the score), at most 768 tokens.
- gpttrace: GPTTrace v8s (Ettin-encoder-1B sequence classification, the repo's model/ folder) on gpttrace_normalize
  text, at most 512 tokens; score = logit(machine) - logit(human), as its model card.
- gpttrace_onnx: GPTTrace Ettin-150M v5e, the int8 ONNX model under onnxruntime (CPU), on gpttrace_normalize text,
  ids beyond 512 cut to the first 511 and the last; score = logit(ai) - logit(human).
Higher scores are more AI-like for all.
"""
import glob
import os
import re
import unicodedata

import numpy as np
import pandas as pd
import torch
from safetensors import safe_open
from safetensors.torch import load_file
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from . import cli, path
from .phase import cell_rng, test_documents
from .pretrain import sentence_table
from .stats import conformal_pvalues, n_edited, sample_truth


class NormedLinear(torch.nn.Module):
    """EditLens's head: LayerNorm, then a linear layer without bias."""

    def __init__(self, hidden, labels):
        super().__init__()
        self.norm, self.linear = torch.nn.LayerNorm(hidden), torch.nn.Linear(hidden, labels, bias=False)

    def forward(self, x):
        return self.linear(self.norm(x))


BOILERPLATE_STARTS = ("Sure", "Here", "Abstract", "Title", "I'm happy to help", "Certainly")


def editlens_clean(text):
    """EditLens's clean_text (scripts/preprocess.py): emoji to names, drop a reasoning block before </think>, drop a
    first line that opens with assistant boilerplate, lowercase, collapse whitespace."""
    import emoji
    text = emoji.demojize(text)
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    paras = [p for p in text.split("\n") if p.strip()]
    if paras:
        first = emoji.replace_emoji(re.sub(r"^[^a-zA-Z0-9]*", "", paras[0]), "")
        if first.startswith(BOILERPLATE_STARTS) and len(paras) > 1:
            text = "\n".join(paras[1:])
    return re.sub(r"\s+", " ", text.lower()).strip()


def editlens(d):
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from transformers import BitsAndBytesConfig
    adapter = snapshot_download(d["model"])
    with safe_open(glob.glob(f"{adapter}/adapter_model.safetensors")[0], framework="np") as f:
        k = next(f.get_slice(n).get_shape()[0] for n in f.keys() if n.endswith("score.linear.weight"))
    tok = AutoTokenizer.from_pretrained(adapter)
    if tok.pad_token is None:
        tok.pad_token, tok.padding_side = tok.eos_token, "left"
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    base = AutoModelForSequenceClassification.from_pretrained(d["base"], num_labels=k, quantization_config=quant,
                                                              device_map={"": 0})
    base.score = NormedLinear(base.config.hidden_size, k).cuda()
    base.config.pad_token_id = tok.pad_token_id
    model = PeftModel.from_pretrained(base, adapter).eval()
    buckets = torch.arange(k, device="cuda", dtype=torch.float32) / (k - 1)

    def score(texts):
        enc = tok([editlens_clean(t) for t in texts], return_tensors="pt", padding=True, truncation=True,
                  max_length=d["max_tokens"]).to("cuda")
        with torch.autocast("cuda", torch.bfloat16):
            logits = model(**enc).logits
        return logits.float().softmax(-1) @ buckets
    return score


def desklib(d):
    from huggingface_hub import snapshot_download
    p = snapshot_download(d["model"])
    tok, cfg = AutoTokenizer.from_pretrained(p), AutoConfig.from_pretrained(p)
    state = load_file(f"{p}/model.safetensors")
    backbone = AutoModel.from_config(cfg)
    backbone.load_state_dict({k[6:]: v for k, v in state.items() if k.startswith("model.")})
    head = torch.nn.Linear(cfg.hidden_size, 1)
    head.load_state_dict({k[11:]: v for k, v in state.items() if k.startswith("classifier.")})
    backbone, head = backbone.cuda().eval(), head.cuda().eval()

    def score(texts):
        enc = tok(list(texts), return_tensors="pt", padding=True, truncation=True,
                  max_length=d["max_tokens"]).to("cuda")
        with torch.autocast("cuda", torch.bfloat16):
            h = backbone(**enc).last_hidden_state.float()
        m = enc["attention_mask"][..., None].float()
        return head((h * m).sum(1) / m.sum(1).clamp(min=1e-9))[:, 0]
    return score


HOMOGLYPHS = dict(zip("аеорсухіјѕһԁԛԝӏ" "АВЕКМНОРСТХІЈЅҮԀԚԜ" "ΑΒΕΖΗΙΚΜΝΟΡΤΥΧ" "ονικρτυχα",   # Cyrillic, Greek
                      "aeopcyxijshdqwl" "ABEKMHOPCTXIJSYDQW" "ABEZHIKMNOPTYX" "ovikptuxa"))
GREEK_CYRILLIC = re.compile("[\u0370-\u03ff\u0400-\u052f]")
LATIN, LETTERS = re.compile("[A-Za-z]"), re.compile(r"[^\W\d_]+")   # LETTERS: runs of letters


def gpttrace_normalize(text):
    """GPTTrace's normalize_text (normalize.py in its model repos, typography folding off as there by default): drop
    format characters and variation selectors; in mostly-Latin text, map Cyrillic and Greek lookalike letters to Latin
    in every word that has no other Cyrillic or Greek letter."""
    t = "".join(c for c in str(text) if unicodedata.category(c) != "Cf" and not "\ufe00" <= c <= "\ufe0f")
    look = sum(c in HOMOGLYPHS for c in t)
    if not look or len(LATIN.findall(t)) <= len(GREEK_CYRILLIC.findall(t)) - look:
        return t
    swap = lambda w: "".join(HOMOGLYPHS.get(c, c) for c in w)
    return LETTERS.sub(lambda m: m.group(0) if GREEK_CYRILLIC.search(swap(m.group(0))) else swap(m.group(0)), t)


def gpttrace(d):
    from huggingface_hub import snapshot_download
    p = os.path.join(snapshot_download(d["model"]), "model")
    tok = AutoTokenizer.from_pretrained(p)
    model = AutoModelForSequenceClassification.from_pretrained(p).cuda().eval()

    def score(texts):
        enc = tok([gpttrace_normalize(t) for t in texts], return_tensors="pt", padding=True, truncation=True,
                  max_length=d["max_tokens"]).to("cuda")
        logits = model(**enc).logits.float()
        return logits[:, 1] - logits[:, 0]
    return score


def gpttrace_onnx(d):
    import onnxruntime as ort
    from huggingface_hub import snapshot_download
    from tokenizers import Tokenizer
    p = snapshot_download(d["model"])
    tok = Tokenizer.from_file(f"{p}/tokenizer.json")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))
    sess = ort.InferenceSession(f"{p}/onnx/model_quantized.onnx", opts, providers=["CPUExecutionProvider"])
    pad = tok.token_to_id("[PAD]") or 0

    def score(texts):
        ids = [e.ids for e in tok.encode_batch([gpttrace_normalize(t) for t in texts])]
        ids = [x if len(x) <= d["max_tokens"] else x[:d["max_tokens"] - 1] + x[-1:] for x in ids]
        width = max(map(len, ids))
        x = np.array([i + [pad] * (width - len(i)) for i in ids], np.int64)
        mask = np.array([[1] * len(i) + [0] * (width - len(i)) for i in ids], np.int64)
        feeds = {i.name: mask if "mask" in i.name else x for i in sess.get_inputs()}
        out = sess.run(None, feeds)[0]
        return torch.from_numpy(out[:, 1] - out[:, 0])
    return score


LOADERS = {"editlens": editlens, "desklib": desklib, "gpttrace": gpttrace, "gpttrace_onnx": gpttrace_onnx}


@torch.no_grad()
def score_all(score, texts, bs, parts, chunk=32768):
    """Scores in input order. Length-sorted chunks of `chunk` texts are saved to parts/ as they finish, so a
    preempted run resumes at the first missing one."""
    order = np.argsort([len(t) for t in texts], kind="stable")   # length-sorted batches waste less padding
    out = np.empty(len(texts))
    for k, start in enumerate(tqdm(range(0, len(texts), chunk))):
        f, idx = parts / f"{k:04d}.npy", order[start:start + chunk]
        if not f.exists():
            np.save(parts / "tmp.npy", np.concatenate([score([texts[j] for j in idx[i:i + bs]]).float().cpu().numpy()
                                                       for i in range(0, len(idx), bs)]))
            (parts / "tmp.npy").replace(f)
        out[idx] = np.load(f)
    return out


def documents(cfg, versions, sents):
    """Rows (source, doc_id, split, editor, theta, n, m, text) of the calib null documents and phase.py's whole-mode
    test documents."""
    c, tab = cfg["phase"], sentence_table(sents)
    text_of = dict(zip(versions.version_id, versions.text))
    rows = []
    for d, g in tab[tab.split == "calib"].groupby("doc_id"):
        if len(g) >= c["whole_min_sentences"]:
            rows.append((g.source.iat[0], d, "calib", "none", np.inf, len(g), 0, "".join(g.sep + g.text)))
    seps = dict(zip(tab.sentence_id, tab.sep))
    human_text = dict(zip(tab.sentence_id, tab.text))
    for source in sorted(versions.source.unique()):
        v = versions[versions.source == source]
        groups = test_documents(v, "whole", np.nan, cell_rng(cfg, "window", source, "whole", np.nan),
                                c["whole_min_sentences"])
        doc_of = dict(zip(v.sentence_id, v.doc_id))
        join = lambda ids, swap: "".join(seps[s] + swap.get(s, human_text[s]) for s in ids)
        for ids in groups:
            for row in ids:
                rows.append((source, doc_of[row[0]], "test", "none", np.inf, len(row), 0, join(row, {})))
        edits = v[(v.editor != "human") & v.split.isin(["test", "val"])]
        for (editor, target_r), e in edits.groupby(["editor", "target_r"], dropna=False):
            edit_text = dict(zip(e.sentence_id, e.version_id.map(text_of)))
            for theta in c["thetas"]:
                rng = cell_rng(cfg, "truth", source, "whole", np.nan, editor, target_r, theta)
                for ids in groups:
                    chosen = sample_truth(*ids.shape, n_edited(ids.shape[1], theta), rng)
                    for row, ch in zip(ids, chosen):
                        swap = {s: edit_text[s] for s in row[ch] if s in edit_text}
                        rows.append((source, doc_of[row[0]], "test", editor, theta, len(row), len(swap),
                                     join(row, swap)))
    return pd.DataFrame(rows, columns=["source", "doc_id", "split", "editor", "theta", "n", "m", "text"])


def doc_tests(df, alpha, rng):
    """Each test document's conformal p-value against its source's calib documents (one bin), and rejection."""
    out = []
    for source, g in df.groupby("source"):
        null, test = g[g.split == "calib"], g[g.split == "test"]
        p = conformal_pvalues(null.score.to_numpy(), np.zeros(len(null)), test.score.to_numpy(),
                              np.zeros(len(test)), 1, rng)
        out.append(test.drop(columns="text", errors="ignore").assign(p=p, reject=p <= alpha))
    return pd.concat(out)


def main():
    def args(p):
        p.add_argument("--detectors", nargs="+", help="default: every external.detectors entry")
        g = p.add_mutually_exclusive_group(required=True)
        g.add_argument("--sentences", action="store_true")
        g.add_argument("--documents", action="store_true")
        g.add_argument("--doc-tests", action="store_true")
    cfg, args = cli(__doc__, args)
    c = cfg["external"]
    names = args.detectors or [d["name"] for d in c["detectors"]]
    if args.doc_tests:
        for n in names:
            t = doc_tests(pd.read_parquet(path(cfg, "docscores", f"{n}.parquet")), cfg["phase"]["alpha"],
                          np.random.default_rng(cfg["seed"]))
            t.to_parquet(path(cfg, "docscores", f"{n}_tests.parquet"), index=False)
            print(n, t.groupby(t.editor.eq("none")).reject.mean().to_dict())
        return
    versions = pd.read_parquet(path(cfg, "versions.parquet"),
                               columns=["version_id", "sentence_id", "source", "doc_id", "split", "sent_idx", "editor",
                                        "target_r", "text"])
    if args.documents:
        docs = documents(cfg, versions, pd.read_parquet(path(cfg, "sentences.parquet")))
    for d in c["detectors"]:
        if d["name"] not in names:
            continue
        score = LOADERS[d["kind"]](d)
        if args.sentences:
            v = versions[versions.split.isin(c["splits"])].reset_index(drop=True)
            s = score_all(score, v.text.tolist(), d["batch_size"], path(cfg, "external_parts", d["name"], "x").parent)
            pd.DataFrame({"version_id": v.version_id, "score": s}).to_parquet(
                path(cfg, "scores", f"{d['name']}.parquet"), index=False)
        else:
            s = score_all(score, docs.text.tolist(), d["doc_batch_size"],
                          path(cfg, "external_parts", f"{d['name']}_documents", "x").parent)
            docs.drop(columns="text").assign(score=s).to_parquet(path(cfg, "docscores", f"{d['name']}.parquet"),
                                                                 index=False)
        print(f"{d['name']}: done", flush=True)


if __name__ == "__main__":
    main()
