#!/bin/bash
#SBATCH --job-name=detect
#SBATCH --partition=kempner_requeue
#SBATCH --account=kempner_pehlevan_lab
#SBATCH --constraint=h200
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
# One detector run on one H200: detect.py --run RUN [ARGS]. kempner_requeue preempts without warning and reruns this
# script; detect.py resumes from data/detector/<run>/ckpt.pt (at most ckpt_every steps lost). Models come from the
# HF cache on lab storage (download them on a login node first; on netscratch, which is NFS and was 97% full on
# 2026-11-03, the download hung on its lock files). Usage (from the repo):
#   sbatch --job-name=RUN slurm/detect.sbatch RUN [ARGS]
set -u
source slurm/env.sh
export HF_HOME=/n/holystore01/LABS/pehlevan_lab/Users/ege/huggingface HF_HUB_OFFLINE=1 TQDM_MININTERVAL=60
echo "$(date -u +%FT%TZ) start on $(hostname), restart count ${SLURM_RESTART_COUNT:-0}"
.venv/bin/python -m sparse_edits.detect --run "$@"
