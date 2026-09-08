# MERaLiON vLLM

Local vLLM runner for [MERaLiON-2-10B-ASR](https://huggingface.co/MERaLiON/MERaLiON-2-10B-ASR).

## Run

```bash
uv sync --python 3.12
./scripts/serve-v0.16.sh
python scripts/test-asr.py testdata/librispeech-sample.flac
```

Server listens on `http://127.0.0.1:8000`. Default target is a 16 GiB GPU; it uses online FP8 weights and limits context to 2560 tokens.

## Singapore ASR regression audio

```bash
python scripts/download-mnsc-eval.py
```

Downloads 20 transcripted 5–30s clips from [MERaLiON MNSC v1](https://huggingface.co/datasets/MERaLiON/Multitask-National-Speech-Corpus-v1) into ignored `testdata/mnsc-asr-part1/`. `manifest.jsonl` records source rows and transcripts. Same MERaLiON publisher: use for regression, not independent benchmark claims.

With server running, calculate word error rate (WER):

```bash
python scripts/evaluate-asr.py
```

Writes ignored per-clip predictions to `testdata/mnsc-asr-part1/results.jsonl` and prints aggregate WER.

## Local assets

These are deliberately excluded from Git:

- `models/` — model checkpoints
- `plugins/*/.venv/` — local plugin environments
- `testdata/` — audio samples
- `.venv/` and caches

Put MERaLiON model at `models/MERaLiON/MERaLiON-2-10B-ASR`. Install dependencies from `pyproject.toml`; `vllm-plugin-meralion2==0.3.*` is required for vLLM 0.16.

## vLLM 0.26 experiment

`./scripts/serve-v0.26.sh` starts v0.26 plugin setup on port 8001. Create its ignored environment first:

```bash
(cd plugins/vllm-plugin-meralion2-v026 && uv sync --python 3.12)
```

See [docs/plan.md](docs/plan.md) for FP8 and KV-cache constraints.
