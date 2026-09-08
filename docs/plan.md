# Offline FP8 plan

## Opinion

**Cold start is requirement. Proceed with offline FP8 feasibility work.** It removes runtime weight conversion and should reduce checkpoint I/O, but is not a complete cold-start fix.

Current live server uses 11.08 GiB for FP8 weights; remaining cache is only 0.48 GiB because 16 GiB GPU also runs Xwayland. Offline FP8 does **not** materially reduce GPU weight memory versus current `--quantization fp8`, enlarge KV cache, fix 60-second ASR coverage, or remove FlashInfer compilation.

Observed online weight load is 99.5 s. That is upper-bound time offline FP8 can remove. First boot also included FlashInfer JIT and CUDA graph work; persist those caches and measure a second restart before claiming an end-to-end saving.

Offline FP8 is justified because cold start is now priority. Keep it only if benchmark proves acceptable ASR quality and enough ready-time reduction.

MERaLiON uses custom `MERaLiON2ForConditionalGeneration` code and a vLLM plugin. `vllm-plugin-meralion2==0.3.*` supports `vllm==0.16.*` only. Standard FP8 exporters may not load this architecture. Never hand-cast `.safetensors`; an FP8 checkpoint needs scales and quantization metadata matching vLLM's loader.

## Current baseline

- Server: `vllm==0.16.0`, `vllm-plugin-meralion2==0.3.0`, local FP8-online conversion.
- GPU: RTX 5070 Ti, 16 GiB; Xwayland reserves about 1.2 GiB.
- Weight load: 99.5 s; FP8 model memory: 11.08 GiB.
- Server: `http://127.0.0.1:8000`.
- Safe server setting: `VLLM_MAX_MODEL_LEN=2560`; actual GPU KV cache: 1,488 tokens.
- Smoke tests passed: short audio and 60 s audio. README recommends <=30 s for ASR quality; 60 s request completed but transcribed only part of repeated speech.

## Plan

### 1. Set cold-start target and baseline

Set ready-time SLO first (for example, <120 s). Before conversion, save 10-20 licensed 5-30 s speech clips and expected transcripts. Include English plus target Singapore/SEA languages. Record:

- first and second server-start-to-`/health` time, with persistent caches;
- GPU model/KV memory from server log;
- audio request latency;
- WER or manual transcript comparison;
- output coverage for 60 s audio.

Current `test-meralion.py` and `testdata/librispeech-sample.flac` are smoke tests, not an accuracy suite.

### 2. Preserve caches, then feasibility spike — isolated output only

Keep `~/.cache/vllm`, `~/.cache/flashinfer`, and `~/.cache/torch` on persistent storage. These remove repeat JIT/compile work across process restarts.

Try an official FP8 exporter such as `llm-compressor` against a **copy** of model. First prove it can load MERaLiON remote/custom code and emit a vLLM-recognized FP8 checkpoint. Stop if custom architecture support needs a fork/adapter.

Requirements:

- Preserve source at `models/MERaLiON/MERaLiON-2-10B-ASR`.
- Write output to `models/MERaLiON/MERaLiON-2-10B-ASR-FP8`.
- Reserve roughly 45 GiB free disk: 19 GiB source + ~10 GiB output + conversion temp/cache.
- Run export on >=24 GiB free VRAM or >=24 GiB system RAM. Current host has 16 GiB VRAM and 15 GiB RAM, so it is an inference host, not a safe export host.
- Install exporter only on conversion host: `uv sync --extra quantize --python /usr/bin/python3.12`.
- Export all required model, processor, tokenizer, and quantization config files.
- Use calibration audio representative of deployment if exporter creates static activation scales.

### 3. Validate converted checkpoint

Start same server command against FP8 output, retaining `--quantization fp8` so vLLM selects FP8 checkpoint loader.

Acceptance:

- startup log must **not** select `Fp8OnlineLinearMethod`;
- `/health`, `/v1/models`, and audio chat completion return 200;
- no weight-shape/scale errors;
- 30 s ASR outputs stay within agreed quality threshold of BF16/online-FP8 baseline;
- cold start improves enough to justify extra artifact and conversion maintenance.

### 4. Calibrate FP8 KV for vLLM 0.26

**Status: deferred. Do this on conversion host, not this 16 GiB GPU / 15 GiB RAM host.**

vLLM 0.26 can store KV in FP8 with `--kv-cache-dtype fp8`. Untuned scales default to `1.0`; current v0.26 smoke run logged uncalibrated `q_scale`/`prob_scale` and produced unusable text. This is an observation, not proof that FP8 KV is sole cause: first compare same v0.26 plugin with `--kv-cache-dtype auto`.

Use `llmcompressor` to create a separate calibrated checkpoint; never modify source model in place:

```text
models/MERaLiON/MERaLiON-2-10B-ASR              # source, immutable
models/MERaLiON/MERaLiON-2-10B-ASR-FP8-KV       # calibrated output
```

1. Gather 256--512 representative 5--30 s mono audio clips plus exact production prompt. Include deployment languages. Hold out at least 64 clips with transcripts for WER.
2. On >=24 GiB free VRAM/RAM conversion host, install supported `llmcompressor` version in isolated environment:

   ```bash
   uv sync --extra quantize --python /usr/bin/python3.12
   ```

3. Load source via MERaLiON processor/model remote code. Build calibration examples with that processor so each model call receives both token IDs and audio features. Text-only calibration is invalid for this ASR model.
4. Run `llmcompressor.oneshot` with an FP8 `kv_cache_scheme` using static, symmetric, per-tensor scales. Official recipe shape:

   ```yaml
   kv_cache_scheme:
     num_bits: 8
     type: float
     strategy: tensor
     dynamic: false
     symmetric: true
   ```

   Keep `lm_head` BF16. Do not reuse `--calculate-kv-scales`; vLLM marks it deprecated and it does not provide calibrated q/prob scales.
5. Save compressed model, tokenizer, processor, remote-code files, and quantization config to `MERaLiON-2-10B-ASR-FP8-KV`.
6. Serve calibrated checkpoint on vLLM 0.26 with `--kv-cache-dtype fp8`. Success logs must not contain `Using uncalibrated q_scale 1.0 and/or prob_scale 1.0`.
7. Compare held-out transcript WER, omissions, and hallucinations against BF16-KV control. Reject artifact on regression; only then benchmark cold start and throughput.

Reference: [llm-compressor FP8 KV recipe](https://github.com/vllm-project/llm-compressor/tree/main/examples/quantization_kv_cache).

### 5. Decide

Keep offline FP8 if quality holds and it meets cold-start SLO. Keep server warm for lowest latency; offline FP8 reduces restart time, not request-time cold latency after a process is stopped.

## Not goals

- Offline FP8 will not make 8192 context fit this GPU.
- Offline FP8 will not make 60 s ASR reliable; model guidance is <=30 s for satisfactory ASR.
- Do not change to vLLM 0.26.x without separately porting MERaLiON plugin.
- FP4/NVFP4 is not available in vLLM 0.16. It requires a newer vLLM, which needs the separate MERaLiON plugin port first.
