# Sparse AI edits in human text

Stat 236 project (Çakar, Huang, Nguyen). We AI-edit sentences of pre-2020 human text (web pages, student essays,
news, abstracts, peer reviews, emails, books), score each sentence with several detectors, turn scores into p-values,
and test when sparse edits in a document can be detected and located, compared with the Donoho-Jin rare/weak phase
diagram.

All code and settings are preliminary, only checked to run end to end. Open decisions: [docs/review.md](docs/review.md).
Progress: [docs/to-do.md](docs/to-do.md).

## Layout

- `configs/default.yaml`: every setting.
- `src/sparse_edits/`: one module per step: `corpus` → `edit` → `score_lm` / `score_classifier` → `calibrate` → `phase`.
  `sources.py` loads the human-text datasets; `text.py` and `stats.py` are shared helpers.
- `studies/`: side studies with their own configs (`style_comparison`: which cheap editor edits like a frontier one).
- `tests/`: unit tests.

## Running

```bash
uv sync
uv run pytest
uv run python -m sparse_edits.<step> --help
```

Outputs go under `data_dir` from the config.
