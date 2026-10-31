#!/bin/bash
#SBATCH --job-name=setup-vllm
#SBATCH --partition=serial_requeue
#SBATCH --account=pehlevan_lab
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=slurm_outs/%x_%j.out
# CPU job: the repo env, vLLM in $VENV and GLM 5.3 Flash at its pinned revision (328 GB) in $HF_HOME. Each step resumes
# or does nothing when rerun, so a requeue repeats nothing finished. Usage (from the repo): sbatch slurm/setup_vllm.sbatch
set -eu
source slurm/env.sh
uv sync --frozen
.venv/bin/python -c "import nltk; nltk.download('punkt_tab', quiet=True)"
[ -x $VENV/bin/vllm ] || { uv venv $VENV --python 3.12 --allow-existing \
  && uv pip install --python $VENV/bin/python vllm==0.32.0 --torch-backend=cu130; }
HF_XET_HIGH_PERFORMANCE=1 $VENV/bin/hf download "${GLM[@]}"
echo SETUP_DONE
