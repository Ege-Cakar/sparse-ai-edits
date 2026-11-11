#!/bin/bash
#SBATCH --job-name=thirdparty-segment
#SBATCH --partition=kempner_requeue
#SBATCH --account=kempner_pehlevan_lab
#SBATCH --constraint="h100|h200"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=slurm_outs/%x_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ecakar@college.harvard.edu
# Third-party datasets: thirdparty.py --prepare (CPU) while the corpus segmenter (Qwen3.5-122B-A10B-FP8,
# corpus.segmenter) loads in vLLM 0.32.0, tensor parallel over the job's GPUs (FP8 needs Hopper; 2 H100s leave too
# little memory for its KV cache), then --segment. docs.parquet is kept and segmenter replies are cached in
# data/thirdparty/segment_cache.jsonl, so a rerun (after a preemption or the wall time) repeats only what was in flight.
# Usage (from the repo, through a login shell): sbatch slurm/thirdparty_segment.sbatch (4 x h100|h200, requeue), or
# non-preemptible as decided 2026-11-08: sbatch -p kempner_h200 --constraint=h200 --gpus-per-node=2 --no-requeue ...
set -u
source slurm/env.sh
module load cuda/13.3.1-fasrc01                         # nvcc for vLLM's JIT kernels
export HF_HOME=/n/holystore01/LABS/pehlevan_lab/Users/ege/huggingface HF_HUB_OFFLINE=1 TQDM_MININTERVAL=60 PATH=$VENV/bin:$PATH
C=/n/holystore01/LABS/pehlevan_lab/Users/ege/cache    # caches off /n/home00, which was full on 2026-11-07, seeded with
for d in .triton/cache:triton .cache/vllm:vllm .cache/flashinfer:flashinfer .cache/torch:torch; do   # the kernels the
  rsync -a --ignore-existing ~/${d%%:*}/ $C/${d##*:}/                                 # fresh-corpus run compiled
done
export XDG_CACHE_HOME=$C TRITON_CACHE_DIR=$C/triton
export CPATH=$HOME/.local/share/uv/python/cpython-3.12.12-linux-x86_64-gnu/include/python3.12   # Python.h for triton's
# launchers, which gcc compiles (job 58952815 failed without it): the venv's system Python 3.12.13 has no headers, and
# uv's 3.12.12 has the same C API
export PYTHONPATH=$PWD/src   # the shared .venv installs sparse_edits from the detectors checkout

PORT=$((20000 + SLURM_JOB_ID % 20000))                  # another job on the node may hold 8000
CFG=slurm_outs/config_$SLURM_JOB_ID.yaml
sed "s|localhost:8000|localhost:$PORT|" configs/default.yaml > $CFG
VLLM_USE_DEEP_GEMM=0 $VENV/bin/vllm serve Qwen/Qwen3.5-122B-A10B-FP8 \
  --revision a099dee70ccfcd8d5dda56aaa0b60cb8ecadabc9 --language-model-only --tensor-parallel-size $SLURM_GPUS_ON_NODE \
  --max-model-len 32768 --gpu-memory-utilization 0.90 --max-num-seqs 256 --generation-config vllm --seed 0 \
  --port $PORT >> slurm_outs/vllm_$SLURM_JOB_ID.log 2>&1 &
SERVER=$!
echo "$(date -u +%FT%TZ) start on $(hostname), restart count ${SLURM_RESTART_COUNT:-0}"
[ -f data/thirdparty/docs.parquet ] || .venv/bin/python -m sparse_edits.thirdparty --prepare --config $CFG \
  || { kill $SERVER; exit 1; }
until curl -sf localhost:$PORT/health > /dev/null; do
  kill -0 $SERVER 2> /dev/null || { echo "vLLM exited; see slurm_outs/vllm_$SLURM_JOB_ID.log"; exit 1; }
  sleep 10
done
echo "$(date -u +%FT%TZ) server up on $(hostname):$PORT"
.venv/bin/python -W ignore -m sparse_edits.thirdparty --segment --config $CFG
kill $SERVER
