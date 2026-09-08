#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$ROOT/plugins/vllm-plugin-meralion2-v026/.venv"
export PATH="$VENV/bin:$PATH"
# Endpoint plugins are opt-in. Keep model registry plugin allowlisted too.
export VLLM_PLUGINS="register_meralion2_v026,meralion_whisper_transcription${VLLM_PLUGINS:+,$VLLM_PLUGINS}"
# v0.26's online FP8 weight conversion returns corrupt text on this SM120 GPU.
# 4k BF16 KV needs about 7.1 GiB offload; keep 7.5 GiB safety margin.
export MERALION_CPU_OFFLOAD_GB="${MERALION_CPU_OFFLOAD_GB:-7.5}"
export MERALION_GPU_MEMORY_UTILIZATION="${MERALION_GPU_MEMORY_UTILIZATION:-0.91}"
# BF16 KV is accuracy-safe baseline. Set MERALION_KV_CACHE_DTYPE=fp8 only
# after a calibrated checkpoint passes ASR comparison.
export MERALION_KV_CACHE_DTYPE="${MERALION_KV_CACHE_DTYPE:-auto}"
# 4k covers MERaLiON's 300 s input limit with room for output tokens.
export MERALION_MAX_MODEL_LEN="${MERALION_MAX_MODEL_LEN:-4096}"

exec "$VENV/bin/vllm" serve "$ROOT/models/MERaLiON/MERaLiON-2-10B-ASR" \
  --trust-remote-code \
  --dtype bfloat16 \
  --host 127.0.0.1 \
  --port 8001 \
  --gpu-memory-utilization "$MERALION_GPU_MEMORY_UTILIZATION" \
  --max-model-len "$MERALION_MAX_MODEL_LEN" \
  --kv-cache-dtype "$MERALION_KV_CACHE_DTYPE" \
  --calculate-kv-scales \
  --cpu-offload-gb "$MERALION_CPU_OFFLOAD_GB" \
  --limit-mm-per-prompt '{"audio": 1}' \
  --attention-backend FLASHINFER \
  --enforce-eager # ponytail: skips v0.26's 2.20 GiB CUDA-graph reservation; re-enable when GPU headroom exists
