#!/bin/bash
###############################################################################
# SLURM DIRECTIVES
###############################################################################
#SBATCH --job-name=vllm-server
#SBATCH --partition=scc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -G A100:2
#SBATCH --time=1:00:00
#SBATCH --mem=64G
#SBATCH -c 32
#SBATCH --output=logs/%x-%j.out
#SBATCH -C inet

set -euo pipefail
echo "[$(date +%F\ %T)] Job starting on $HOSTNAME"

###############################################################################
# 1 · Parameters you might tweak
###############################################################################
MODEL="${MODEL:-Qwen/Qwen3-VL-8B-Thinking}"
PORT="${LLM_PORT:-8000}"
TP_SIZE="${SLURM_GPUS_ON_NODE}"
GPU_UTIL="${LLM_GPU_UTIL:-0.90}"
HF_HOME="${HF_HOME:-$SCRATCH/huggingface}"
VENV_VLLM="$HOME/.venv_llm"       # vLLM server env

###############################################################################
# 2 · (Optional) module load / environment setup
###############################################################################
module purge
export HF_HOME; mkdir -p "$HF_HOME" logs
module load miniforge3
source activate vllm

###############################################################################
# 4 · Launch vLLM — write to a temp log file, tail only during startup
###############################################################################
LOG_DIR="$PWD/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/vllm_${SLURM_JOB_ID:-$$}.log"
touch $LOG_FILE

python -u -m vllm.entrypoints.openai.api_server \
       --model "$MODEL" \
       --port "$PORT" \
       --tensor-parallel-size "$TP_SIZE" \
       --gpu-memory-utilization "$GPU_UTIL" \
       --trust-remote-code \
       # --revision "" \
       # --quantization ""
       >"$LOG_FILE" 2>&1 &              # <-- background; output → file only
SERVER_PID=$!

# Show the log *for now* so you can watch weights load
tail -n +1 -f "$LOG_FILE" &
TAIL_PID=$!

cleanup() {
  echo "[$(date +%T)] Cleaning up…"
  kill -SIGINT "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
  kill "$TAIL_PID"   2>/dev/null || true
}
trap cleanup EXIT INT TERM

###############################################################################
# 5 · Health probe (quiet)
###############################################################################
echo -n "⌛ Waiting for vLLM"
for _ in {1..3600}; do
  if curl -s "http://127.0.0.1:${PORT}/v1/models" >/dev/null; then
    echo " … ready."
    break
  fi
  printf "."; sleep 1
done

# ---- server is healthy: stop showing its log ------------------------------
kill "$TAIL_PID" 2>/dev/null || true   # server keeps running, log stays on disk
echo "[`date +%T`] Server output silenced (see $LOG_FILE if needed)"

echo "[$(date +%F\ %T)] vLLM server running on port $PORT. Logs: $LOG_FILE"


################################################################################
echo "starting test python file"
python test_vllm.py
echo "ending test python file"

sleep infinity