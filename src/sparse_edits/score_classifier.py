"""Supervised sentence scores: fine-tuned encoder classifier and sentence embeddings + logistic regression.

Training data: train-split human sentences (label 0) plus `edits_per_human` random accepted edits of each from
non-held-out editors (label 1). The fine-tuned model early-stops on the val split built the same way.
Both score every version and write scores/<name>.parquet (version_id, score = log-odds of "edited").
"""
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegressionCV
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer,
                          TrainingArguments)

from . import cli, path


def labeled(cfg, versions, split):
    v = versions[versions.split == split]
    # Held-out editors test generalization; control editors (train_classifier: false, e.g. masked infilling)
    # are not AI style, so neither is used to train the detector.
    heldout = [e["name"] for e in cfg["edit"]["editors"] if e["heldout"] or not e.get("train_classifier", True)]
    edits = v[(v.editor != "human") & ~v.editor.isin(heldout)]
    edits = edits.groupby("sentence_id").sample(n=cfg["classifier"]["edits_per_human"], random_state=cfg["seed"])
    df = pd.concat([v[v.editor == "human"].assign(label=0), edits.assign(label=1)])
    return df.sample(frac=1, random_state=cfg["seed"])


def save(cfg, name, versions, scores):
    pd.DataFrame({"version_id": versions.version_id, "score": scores}).to_parquet(
        path(cfg, "scores", f"{name}.parquet"), index=False)


def finetune(cfg, versions):
    c = cfg["classifier"]["finetune"]
    tok = AutoTokenizer.from_pretrained(c["model"])
    model = AutoModelForSequenceClassification.from_pretrained(c["model"], num_labels=2)
    encode = lambda df: Dataset.from_pandas(df[["text", "label"]], preserve_index=False).map(
        lambda b: tok(b["text"], truncation=True, max_length=c["max_tokens"]), batched=True)
    args = TrainingArguments(
        output_dir=str(Path(cfg["data_dir"], "models", c["name"])), learning_rate=c["lr"],
        num_train_epochs=c["epochs"], per_device_train_batch_size=c["batch_size"],
        per_device_eval_batch_size=c["batch_size"], eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="eval_loss", save_total_limit=1,
        bf16=torch.cuda.is_available(), seed=cfg["seed"], report_to="none")
    trainer = Trainer(model=model, args=args, data_collator=DataCollatorWithPadding(tok),
                      train_dataset=encode(labeled(cfg, versions, "train")),
                      eval_dataset=encode(labeled(cfg, versions, "val")))
    trainer.train()
    logits = trainer.predict(encode(versions.assign(label=0))).predictions
    save(cfg, c["name"], versions, logits[:, 1] - logits[:, 0])


def embedding_logreg(cfg, versions):
    c = cfg["classifier"]["embedding"]
    encoder = SentenceTransformer(c["model"])
    embed = lambda texts: encoder.encode(list(texts), batch_size=c["batch_size"], normalize_embeddings=True,
                                         show_progress_bar=True)
    train = labeled(cfg, versions, "train")
    clf = LogisticRegressionCV(Cs=10, cv=5, max_iter=2000, random_state=cfg["seed"])
    clf.fit(embed(train.text), train.label)
    save(cfg, c["name"], versions, clf.decision_function(embed(versions.text)))


def main():
    cfg, args = cli(__doc__, lambda p: p.add_argument("--which", choices=["finetune", "embedding"], required=True))
    versions = pd.read_parquet(path(cfg, "versions.parquet"))
    {"finetune": finetune, "embedding": embedding_logreg}[args.which](cfg, versions)


if __name__ == "__main__":
    main()
