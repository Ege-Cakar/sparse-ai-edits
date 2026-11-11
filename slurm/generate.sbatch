#!/bin/bash
#SBATCH --job-name=generate
#SBATCH --partition=kempner_requeue
#SBATCH --account=kempner_pehlevan_lab
#SBATCH --constraint=h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=slurm_outs/%x_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ecakar@college.harvard.edu
# One open model served by vLLM 0.32.0 with the settings the RunPod editors used (tensor-parallel over the job's GPUs),
# then generate.py with the remaining arguments, rerun until it has no document left to send (up to 5 rounds).
# kempner_requeue preempts without warning and reruns this script; generate.py skips finished documents and the
# segmenter's reply cache keeps finished chunks, so a rerun repeats only what was in flight. Usage (from the repo):
#   sbatch --job-name=generate-gemma slurm/generate.sbatch gemma --generators gemma-4-31b
#   sbatch --job-name=segment --gpus-per-node=2 slurm/generate.sbatch qwen --segment
set -u
source slurm/env.sh
module load cuda/13.3.1-fasrc01                         # nvcc for vLLM's JIT kernels
export HF_HOME=/n/holystore01/LABS/pehlevan_lab/Users/ege/huggingface HF_HUB_OFFLINE=1 PATH=$VENV/bin:$PATH \
  TQDM_MININTERVAL=60 PYTHONPATH=$PWD/src            # this checkout's code, even with another checkout's .venv
PORT=$((20000 + SLURM_JOB_ID % 20000))                  # another job on the node may hold 8000
CFG=slurm_outs/config_$SLURM_JOB_ID.yaml
sed "s|localhost:8000|localhost:$PORT|" configs/default.yaml > $CFG
COMMON="--tensor-parallel-size $SLURM_GPUS_ON_NODE --max-model-len 32768 --gpu-memory-utilization 0.90 \
  --max-num-seqs 256 --generation-config vllm --seed 0 --port $PORT"
case $1 in
  gemma) MODEL="google/gemma-4-31B-it --revision 842da3794eaa0b77d5f08bae87a17459d91ff475 --language-model-only" ;;
  gptoss) MODEL="openai/gpt-oss-120b --revision b5c939de8f754692c1647ca79fbf85e8c1e70f8a"
          export VLLM_SYSTEM_START_DATE=2026-11-01 ;;   # the harmony system message's date, as edit.SYSTEM's
  qwen) MODEL="Qwen/Qwen3.5-122B-A10B-FP8 --revision a099dee70ccfcd8d5dda56aaa0b60cb8ecadabc9 --language-model-only"
        export VLLM_USE_DEEP_GEMM=0 ;;
  *) echo "unknown server $1"; exit 1 ;;
esac
shift
vllm serve $MODEL $COMMON >> slurm_outs/vllm_$SLURM_JOB_ID.log 2>&1 &
SERVER=$!
until curl -sf localhost:$PORT/health > /dev/null; do
  kill -0 $SERVER 2> /dev/null || { echo "vLLM exited; see slurm_outs/vllm_$SLURM_JOB_ID.log"; exit 1; }
  sleep 10
done
echo "$(date -u +%FT%TZ) server up on $(hostname):$PORT, restart count ${SLURM_RESTART_COUNT:-0}"
for round in 1 2 3 4 5; do
  .venv/bin/python -W ignore -m sparse_edits.generate --config $CFG "$@" 2>&1 | tee slurm_outs/generate_$SLURM_JOB_ID.last
  echo "$(date -u +%FT%TZ) round $round done"
  grep -q "[1-9][0-9]* documents to send" slurm_outs/generate_$SLURM_JOB_ID.last || break
done
kill $SERVER
