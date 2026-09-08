#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
export PATH="$PWD/.venv/bin:$PATH"

# FP8 weights + KV cache. Xwayland reserves ~1.2 GiB of this 16 GiB GPU,
# so model-config auto (8192) cannot initialize. Override context only after test.
export VLLM_CPU_OFFLOAD_GB="${VLLM_CPU_OFFLOAD_GB:-0}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.91}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-2560}"

# ponytail: test server; tune offload, context, and memory after successful load.
exec .venv/bin/vllm serve models/MERaLiON/MERaLiON-2-10B-ASR \
  --trust-remote-code \
  --dtype bfloat16 \
  --quantization fp8 \
  --host 127.0.0.1 \
  --port 8000 \
  --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
  --max-model-len "$VLLM_MAX_MODEL_LEN" \
  --kv-cache-dtype fp8 \
  --calculate-kv-scales \
  --cpu-offload-gb "$VLLM_CPU_OFFLOAD_GB" \
  --limit-mm-per-prompt '{"audio": 1}' \
  --attention-backend FLASHINFER
