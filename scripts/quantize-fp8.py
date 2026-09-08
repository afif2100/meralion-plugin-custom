#!/usr/bin/env python3
"""Export MERaLiON BF16 checkpoint as vLLM-compatible offline FP8."""

import argparse
import os
import shutil
from pathlib import Path

import torch
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

MIN_RAM_GIB = 24


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("models/MERaLiON/MERaLiON-2-10B-ASR"))
    parser.add_argument("--output", type=Path, default=Path("models/MERaLiON/MERaLiON-2-10B-ASR-FP8"))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()

    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    if shutil.disk_usage(".").free < 25 * 2**30:
        raise SystemExit("need 25 GiB free disk for source, output, and temp files")
    if args.device == "cuda":
        if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 24 * 2**30:
            raise SystemExit("need >= 24 GiB free GPU VRAM for GPU export; use a larger GPU")
    elif os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") < MIN_RAM_GIB * 2**30:
        raise SystemExit(f"need >= {MIN_RAM_GIB} GiB RAM for CPU export")

    load_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
    }
    if args.device == "cuda":
        load_kwargs["device_map"] = "cuda"
    model = AutoModelForSpeechSeq2Seq.from_pretrained(args.model, **load_kwargs)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model.eval()

    # FP8_DYNAMIC: static per-channel weights, dynamic per-token activations.
    # No calibration data needed. Keep lm_head BF16 for transcription quality.
    oneshot(
        model=model,
        recipe=QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"]),
    )

    args.output.mkdir(parents=True)
    model.save_pretrained(args.output, safe_serialization=True)
    processor.save_pretrained(args.output)
    for source in args.model.glob("*.py"):
        shutil.copy2(source, args.output / source.name)
    print(args.output)


if __name__ == "__main__":
    main()
