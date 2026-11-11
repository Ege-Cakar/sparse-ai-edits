#!/bin/bash
#SBATCH --job-name=thirdparty-score
#SBATCH --partition=kempner_requeue
#SBATCH --account=kempner_pehlevan_lab
#SBATCH --constraint="h100|h200|a100"   # full-olmo-3-32b (64 GB in bf16): submit with --constraint="h100|h200"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=slurm_outs/%x_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ecakar@college.harvard.edu
# One detector on the third-party datasets on one GPU: thirdparty.py --score NAME (a thirdparty.detectors name).
# kempner_requeue preempts without warning and reruns this script; scores are cached by text hash, so it resumes.
# Usage (from the repo): sbatch --job-name=thirdparty-NAME slurm/thirdparty_score.sbatch NAME
set -u
source slurm/env.sh
export HF_HOME=/n/holystore01/LABS/pehlevan_lab/Users/ege/huggingface HF_HUB_OFFLINE=1 TQDM_MININTERVAL=60
export XDG_CACHE_HOME=/n/holystore01/LABS/pehlevan_lab/Users/ege/cache TRITON_CACHE_DIR=/n/holystore01/LABS/pehlevan_lab/Users/ege/cache/triton   # /n/home00 was full on 2026-11-07
export PYTHONPATH=$PWD/src   # the shared .venv installs sparse_edits from the detectors checkout
echo "$(date -u +%FT%TZ) start on $(hostname), restart count ${SLURM_RESTART_COUNT:-0}"
.venv/bin/python -m sparse_edits.thirdparty --score "$1"
