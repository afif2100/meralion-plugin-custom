#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$ROOT/.venv"
export PATH="$VENV/bin:$PATH"

# One request at a time: benchmark setting for 16 GiB GPU.
exec "$VENV/bin/meralion-3-asr" serve \
  --model "${MERALION_V3_MODEL:-$ROOT/models/MERaLiON/MERaLiON-3-3B-ASR}" \
  --host 127.0.0.1 \
  --port "${MERALION_V3_PORT:-8001}" \
  --gpu-memory-utilization "${MERALION_GPU_MEMORY_UTILIZATION:-0.90}" \
  --max-model-len "${MERALION_MAX_MODEL_LEN:-1300}" \
  --max-num-seqs 1 \
  --attention-backend FLASHINFER
