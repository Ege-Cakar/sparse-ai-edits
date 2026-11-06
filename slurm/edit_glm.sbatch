#!/bin/bash
#SBATCH --job-name=edit-glm
#SBATCH --partition=kempner_requeue
#SBATCH --account=kempner_pehlevan_lab
#SBATCH --constraint=h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=384G
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=slurm_outs/%x_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ecakar@college.harvard.edu
# GLM 5.3 Flash on one node's 4 H200s, with the RunPod server settings: start vLLM, run edit.py for each named editor
# until no request is left (up to 5 rounds), stop. kempner_requeue preempts without warning and reruns this script;
# edit.py writes each finished record as a line and skips finished sentences, so a rerun repeats only the requests that
# were in flight, after a ~10 min server start. Usage (from the repo): sbatch slurm/edit_glm.sbatch EDITOR...
set -u
source slurm/env.sh
module load cuda/13.3.1-fasrc01                         # nvcc for vLLM's JIT kernels
export HF_HUB_OFFLINE=1 PATH=$VENV/bin:$PATH TQDM_MININTERVAL=60
PORT=$((20000 + SLURM_JOB_ID % 20000))                  # another job on the node may hold 8000
CFG=slurm_outs/config_$SLURM_JOB_ID.yaml
sed "s|localhost:8000|localhost:$PORT|" configs/default.yaml > $CFG
vllm serve "${GLM[@]}" --tensor-parallel-size 4 --reasoning-parser glm47 --language-model-only --max-model-len 32768 \
  --gpu-memory-utilization 0.90 --max-num-seqs 256 --generation-config vllm --seed 0 --port $PORT \
  >> slurm_outs/vllm_$SLURM_JOB_ID.log 2>&1 &
SERVER=$!
until curl -sf localhost:$PORT/health > /dev/null; do
  kill -0 $SERVER 2> /dev/null || { echo "vLLM exited; see slurm_outs/vllm_$SLURM_JOB_ID.log"; exit 1; }
  sleep 10
done
echo "$(date -u +%FT%TZ) server up on $(hostname):$PORT"
for ed in "$@"; do
  for round in 1 2 3 4 5; do
    .venv/bin/python -W ignore -m sparse_edits.edit --config $CFG --editors $ed
    left=$(.venv/bin/python -W ignore -m sparse_edits.edit --config $CFG --editors $ed --dry-run | grep -o "[0-9]* requests to send" | cut -d' ' -f1)
    echo "$(date -u +%FT%TZ) $ed round $round: $left left"
    [ "$left" = "0" ] && break
  done
done
kill $SERVER
