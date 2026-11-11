# Who Touched My Sentence?

Code for the Stat 236 project "Who Touched My Sentence? Inference for AI-Edited Documents in the Sparse and Weak
Regime" (Çakar, Huang, Nguyen). LLMs edit sentences of human documents, sentence detectors are fine-tuned on the edits,
their scores become conformal p-values, and Higher Criticism tests each document for AI edits. Methods and results are
in the paper, [`report/report.pdf`](report/report.pdf); the numbers behind its tables are in [`results/`](results/).

## Setup

```bash
uv sync
```

API editors read `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` from `.env`. Open-weight editors and the sentence segmenter
call a vLLM server at `localhost:8000` (`slurm/setup_vllm.sbatch`). Gated Hugging Face models and datasets (Llama 3.x,
EditLens) need a Hugging Face login.

## Running

Each step is a module in `src/sparse_edits/` that reads `configs/default.yaml` and writes under `data/`. In order:
`corpus` → `edit` → `detect` → `ensemble` → `calibrate` → `phase` → `report`.

```bash
uv run python -m sparse_edits.corpus --help
uv run pytest
```

GPU steps have Slurm scripts in `slurm/`, written for the Harvard cluster.
