#!/bin/bash
#SBATCH --job-name=continued-pretrain
#SBATCH --partition=kempner_requeue
#SBATCH --account=kempner_pehlevan_lab
#SBATCH --constraint=h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=slurm_outs/%x_%j.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ecakar@college.harvard.edu
# Continued pretraining with LoRA on one H200: continued_pretrain.py [--mode base|sweep|train] [--bases ...]
# [--arms ...] [--seeds ...]. kempner_requeue preempts without warning and reruns this script; a run resumes from its
# ckpt.pt and finished runs (metrics.json) are skipped, so concurrent jobs must not share runs (split them by base and
# seed). Models come from the HF cache on lab storage. Usage (from the repo):
#   sbatch --job-name=cpt-BASE-seedS slurm/continued_pretrain.sbatch --bases BASE --seeds S
set -u
source slurm/env.sh
export HF_HOME=/n/holystore01/LABS/pehlevan_lab/Users/ege/huggingface HF_HUB_OFFLINE=1
export PYTHONPATH=$PWD/src   # this checkout's code, also when .venv is another checkout's (installed editable there)
echo "$(date -u +%FT%TZ) start on $(hostname), restart count ${SLURM_RESTART_COUNT:-0}"
.venv/bin/python -m sparse_edits.continued_pretrain "$@"
