#!/bin/bash
#SBATCH --job-name=thirdparty-report
#SBATCH --partition=test
#SBATCH --account=pehlevan_lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --open-mode=append
#SBATCH --output=slurm_outs/%x_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ecakar@college.harvard.edu
# CPU-only: thirdparty_report.py's tables (data/thirdparty/report.md and report/*.csv) once thirdparty_eval.py has
# run for every detector. Usage (from the repo, through a login shell): sbatch slurm/thirdparty_report.sbatch
set -u
source slurm/env.sh
export PYTHONPATH=$PWD/src OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "$(date -u +%FT%TZ) start on $(hostname)"
.venv/bin/python -m sparse_edits.thirdparty_report
