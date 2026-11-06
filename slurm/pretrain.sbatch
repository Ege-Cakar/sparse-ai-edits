#!/bin/bash
#SBATCH --job-name=pretrain
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
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ecakar@college.harvard.edu
# Pretraining runs on one H200: pretrain.py [--arms ...] [--sizes ...] [--seeds ...]. kempner_requeue preempts without
# warning and reruns this script; finished runs (metrics.json) are skipped and an unfinished one restarts (runs take
# minutes). The tokenizer comes from the HF cache on lab storage. Usage (from the repo):
#   sbatch --job-name=pretrain-ARM slurm/pretrain.sbatch --arms ARM
set -u
source slurm/env.sh
export HF_HOME=/n/holystore01/LABS/pehlevan_lab/Users/ege/huggingface HF_HUB_OFFLINE=1
echo "$(date -u +%FT%TZ) start on $(hostname), restart count ${SLURM_RESTART_COUNT:-0}"
.venv/bin/python -m sparse_edits.pretrain "$@"
