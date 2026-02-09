#!/bin/bash
#SBATCH --job-name=vllm-server
#SBATCH --partition=scc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -G A100:4
#SBATCH --time=14:00:00
#SBATCH --mem=128G
#SBATCH -c 32
#SBATCH --output=logs/%x-%j.out
#SBATCH -C inet

set -euo pipefail
echo "[$(date +%F\ %T)] Job starting on $HOSTNAME"
# MODEL="${MODEL:-Qwen/Qwen3-VL-2B-Instruct}"
# MODEL="${MODEL:-Qwen/Qwen3-VL-8B-Thinking}"
# MODEL="${MODEL:-Qwen/Qwen3-VL-32B-Thinking}"
# MODEL="${MODEL:-Qwen/Qwen3-VL-32B-Instruct}"
MODEL="${MODEL:-Qwen/Qwen3-VL-32B-Thinking}"
# PORT="${LLM_PORT:-8000}"
PORT="${LLM_PORT:-$((8000 + (SLURM_JOB_ID % 1000)))}"
TP_SIZE="${SLURM_GPUS_ON_NODE}"
GPU_UTIL="${LLM_GPU_UTIL:-0.95}"
HF_HOME="${HF_HOME:-$SCRATCH/huggingface}"
VENV_VLLM="$HOME/.venv_llm"       # vLLM server env

module purge
export HF_HOME; mkdir -p "$HF_HOME" logs
module load miniforge3
source activate vllm

LOG_DIR="$PWD/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/vllm_${SLURM_JOB_ID:-$$}.log"
touch $LOG_FILE

# ---- evaluation argument env vars (override as needed) ---------------------
MODEL_SHA="${MODEL_SHA:-}"               # empty → auto-resolve to mapping/latest
BOARD_TYPE="${BOARD_TYPE:-original}"
PROMPT_TYPE="${PROMPT_TYPE:-default_tr}"
SUBSET="${SUBSET:-all}"
SPLIT="${SPLIT:-test}"
TEMPERATURE="${TEMPERATURE:-0.6}"
MAX_TOKENS="${MAX_TOKENS:-81920}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"
MAX_CONCURRENT_REQUESTS="${MAX_CONCURRENT_REQUESTS:-200}"
SEED="${SEED:-42}"

echo "Starting Vllm Api"

python -u -m vllm.entrypoints.openai.api_server \
       --model "$MODEL" \
       --port "$PORT" \
       --tensor-parallel-size "$TP_SIZE" \
       --gpu-memory-utilization "$GPU_UTIL" \
       --trust-remote-code \
       >"$LOG_FILE" 2>&1 &              # server background
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
  printf "."; sleep 1
done

echo "✔ vLLM ready. Last 20 log lines:"
tail -n 20 "$LOG_FILE"

echo "[$(date +%F\ %T)] vLLM server running on port $PORT. Logs: $LOG_FILE"

source activate sparcVisualPuzzleEval
# Run evaluation with explicit CLI arguments
echo "Starting Python Script"

python evaluate.py \
  --model "$MODEL" \
  --model-sha "${MODEL_SHA:-}" \
  --board-type "$BOARD_TYPE" \
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

