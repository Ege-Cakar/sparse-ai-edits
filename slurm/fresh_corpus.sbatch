#!/bin/bash
#SBATCH --job-name=fresh-corpus
#SBATCH --partition=kempner_requeue
#SBATCH --account=kempner_pehlevan_lab
#SBATCH --constraint=h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=slurm_outs/%x_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ecakar@college.harvard.edu
# Fresh documents for the pretraining fresh-data control: the corpus segmenter (Qwen3.5-122B-A10B-FP8, corpus.segmenter)
# served by vLLM 0.32.0 on this job's 2 H200s, then fresh.py, which streams the sources (network) and writes
# data/fresh/sentences.parquet. kempner_requeue preempts without warning and reruns this script; segmenter replies
# are cached in data/fresh/segment_cache.jsonl, so a rerun repeats only what was in flight. Usage (from the repo):
#   sbatch slurm/fresh_corpus.sbatch
set -u
source slurm/env.sh
module load cuda/13.3.1-fasrc01                         # nvcc for vLLM's JIT kernels
export HF_HOME=/n/holystore01/LABS/pehlevan_lab/Users/ege/huggingface TQDM_MININTERVAL=60
PORT=$((20000 + SLURM_JOB_ID % 20000))                  # another job on the node may hold 8000
CFG=slurm_outs/config_$SLURM_JOB_ID.yaml
sed "s|localhost:8000|localhost:$PORT|" configs/default.yaml > $CFG
HF_HUB_OFFLINE=1 VLLM_USE_DEEP_GEMM=0 $VENV/bin/vllm serve Qwen/Qwen3.5-122B-A10B-FP8 \
  --revision a099dee70ccfcd8d5dda56aaa0b60cb8ecadabc9 --language-model-only --tensor-parallel-size 2 \
  --max-model-len 32768 --gpu-memory-utilization 0.90 --max-num-seqs 256 --generation-config vllm --seed 0 \
  --port $PORT >> slurm_outs/vllm_$SLURM_JOB_ID.log 2>&1 &
SERVER=$!
until curl -sf localhost:$PORT/health > /dev/null; do
  kill -0 $SERVER 2> /dev/null || { echo "vLLM exited; see slurm_outs/vllm_$SLURM_JOB_ID.log"; exit 1; }
  sleep 10
done
echo "$(date -u +%FT%TZ) server up on $(hostname):$PORT, restart count ${SLURM_RESTART_COUNT:-0}"
.venv/bin/python -W ignore -m sparse_edits.fresh --config $CFG
kill $SERVER
