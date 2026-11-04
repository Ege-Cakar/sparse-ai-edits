#!/bin/bash
#SBATCH --job-name=score-lm
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
# Zero-shot scores of one observer/performer pair on one H200: score_lm.py --pair PAIR. kempner_requeue preempts
# without warning and reruns this script; score_lm.py skips the chunks already in data/lm_score_parts/PAIR/.
# Models come from the HF cache on lab storage (download them on a login node first). Usage (from the repo):
#   sbatch --job-name=score-lm-PAIR slurm/score_lm.sbatch PAIR
set -u
source slurm/env.sh
export HF_HOME=/n/holystore01/LABS/pehlevan_lab/Users/ege/huggingface HF_HUB_OFFLINE=1 TQDM_MININTERVAL=60
echo "$(date -u +%FT%TZ) start on $(hostname), restart count ${SLURM_RESTART_COUNT:-0}"
.venv/bin/python -m sparse_edits.score_lm --pair "$1"
