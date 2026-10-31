# Harvard FASRC paths for the sbatch scripts; sourced from the repo root.
export UV_CACHE_DIR=/n/holystore01/LABS/pehlevan_lab/Users/ege/.uv-cache
export HF_HOME=/n/netscratch/pehlevan_lab/Lab/ecakar/huggingface
VENV=/n/holystore01/LABS/pehlevan_lab/Users/ege/vllm-0.32.0   # vLLM 0.32.0 with torch cu130, as on the RunPod editors
GLM=(zai-org/GLM-5.3-Flash --revision 46a050bec13002ad29ee795c16e61cf4fa3c4a9c)
module load gcc/13.2.0-fasrc01   # vLLM's JIT kernels need C++17; Rocky 8's gcc 8 defaults to C++14 (RunPod had gcc 13)
