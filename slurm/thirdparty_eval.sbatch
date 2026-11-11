#!/bin/bash
#SBATCH --job-name=thirdparty-eval
#SBATCH --partition=shared
#SBATCH --account=pehlevan_lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --open-mode=append
#SBATCH --output=slurm_outs/%x_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ecakar@college.harvard.edu
# CPU-only analysis of the third-party datasets (thirdparty_eval.py ARGS), on shared under pehlevan_lab (Ege,
# 2026-11-10: no GPU needed). Usage (from the repo, through a login shell):
#   sbatch --job-name=thirdparty-eval-NAME slurm/thirdparty_eval.sbatch --detectors NAME
set -u
source slurm/env.sh
export PYTHONPATH=$PWD/src OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK \
  MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "$(date -u +%FT%TZ) start on $(hostname)"
.venv/bin/python -m sparse_edits.thirdparty_eval "$@"
