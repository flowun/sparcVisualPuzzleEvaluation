#!/bin/bash
#SBATCH --job-name=rules-vllm-server
#SBATCH --partition=scc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -G A100:4
#SBATCH --time=48:00:00
#SBATCH --mem=128G
#SBATCH -c 32
#SBATCH --output=logs/%x-%j.out
#SBATCH -C inet

set -euo pipefail
echo "[$(date +%F\ %T)] Job starting on $HOSTNAME"
EVAL_SCRIPT="${EVAL_SCRIPT:-evaluate_object_detection.py}"
# MODEL="${MODEL:-Qwen/Qwen3-VL-2B-Instruct}"
# MODEL="${MODEL:-Qwen/Qwen3-VL-8B-Thinking}"
# MODEL="${MODEL:-Qwen/Qwen3-VL-32B-Thinking}"
# MODEL="${MODEL:-Qwen/Qwen3-VL-32B-Instruct}"
# MODEL="${MODEL:-zai-org/GLM-4.6V}"
# MODEL="${MODEL:-Qwen/Qwen3.5-27B}"
# MODEL="${MODEL:-mistralai/Mistral-Small-3.2-24B-Instruct-2506}"
MODEL="${MODEL:-meta-llama/Llama-4-Scout-17B-16E-Instruct}"
# MODEL="${MODEL:-google/gemma-3-27b-it}"
# MODEL="${MODEL:-google/gemma-4-31B-it}"
# MODEL="${MODEL:-QuantTrio/Qwen3.5-397B-A17B-AWQ}"
# PORT="${LLM_PORT:-8000}"
PORT="${LLM_PORT:-$((8000 + (SLURM_JOB_ID % 1000)))}"
TP_SIZE="${SLURM_GPUS_ON_NODE}"
GPU_UTIL="${LLM_GPU_UTIL:-0.95}"
HF_HOME="/projects/scc/UGOE/UXEI/UMIN/ag_gipp/dir.project/lars/hf"

module purge
module load gcc cuda apptainer
export HF_HOME
mkdir -p "$HF_HOME" logs

LOG_DIR="$PWD/logs"
mkdir -p "$LOG_DIR"

# ---- evaluation argument env vars (override as needed) ---------------------
MODEL_SHA="${MODEL_SHA:-}" # empty → auto-resolve to mapping/latest
BOARD_TYPE="${BOARD_TYPE:-original}"
PROMPT_TYPE="${PROMPT_TYPE:-default}"
SUBSET="${SUBSET:-all}"
SPLIT="${SPLIT:-test}"
TEMPERATURE="${TEMPERATURE:-0.6}"
MAX_TOKENS="${MAX_TOKENS:-81920}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"
MAX_CONCURRENT_REQUESTS="${MAX_CONCURRENT_REQUESTS:-200}"
SEED="${SEED:-42}"

echo "Starting Vllm Api"

apptainer exec --cleanenv --env "HF_HOME=$HF_HOME" -B $HF_HOME:$HF_HOME:rw --nv /projects/scc/UGOE/UXEI/UMIN/ag_gipp/dir.project/vllm-container/vllm_gemma4.sif vllm serve "$MODEL" \
  --port "$PORT" \
  --tensor-parallel-size "$TP_SIZE" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --trust-remote-code \
  --max-model-len 100000 &

SERVER_PID=$!

cleanup() {
  echo "[$(date +%T)] Cleaning up…"
  kill -SIGINT "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# health probe
echo -n "⌛ Waiting for vLLM"
for _ in {1..3600}; do
  if curl -s "http://127.0.0.1:${PORT}/v1/models" >/dev/null; then
    echo " … ready."
    break
  fi
  printf "."
  sleep 1
done

echo "✔ vLLM ready. Last 20 log lines:"

echo "[$(date +%F\ %T)] vLLM server running on port $PORT."

source .venv/bin/activate
# Run evaluation with explicit CLI arguments
echo "Starting Python Script"

# Loop over the board types we want to evaluate
BOARD_TYPES=("original" "coordinate_grid" "start_end_marked" "coordinate_grid_and_start_end_marked" "path_cell_annotated" "text")

for BT in "${BOARD_TYPES[@]}"; do
  echo "Evaluating board type: $BT"
  python "$EVAL_SCRIPT" \
    --model "$MODEL" \
    --model-sha "${MODEL_SHA:-}" \
    --board-type "$BT" \
    --prompt-type "$PROMPT_TYPE" \
    --subset "$SUBSET" \
    --split "$SPLIT" \
    --api-port "$PORT" \
    --temperature "$TEMPERATURE" \
    --max-tokens "$MAX_TOKENS" \
    --top-p "$TOP_P" \
    --top-k "$TOP_K" \
    --max-concurrent-requests "$MAX_CONCURRENT_REQUESTS" \
    --seed "$SEED"
done
