#!/usr/bin/env python3
"""Download a small Singapore-English ASR regression set from MNSC v1."""

import argparse
import json
import shutil
import time
import wave
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import urlopen

DATASET = "MERaLiON/Multitask-National-Speech-Corpus-v1"
CONFIG = "ASR-PART1-Test"
SPLIT = "train"
API = "https://datasets-server.huggingface.co/rows"
ROOT = Path(__file__).resolve().parent.parent


def rows(offset: int, length: int = 20) -> dict:
    query = urlencode({"dataset": DATASET, "config": CONFIG, "split": SPLIT, "offset": offset, "length": length})
    for wait in (1, 5, 15):
        try:
            with urlopen(f"{API}?{query}", timeout=60) as response:
                return json.load(response)
        except HTTPError as error:
            if error.code != 429:
                raise
            time.sleep(wait)
    raise SystemExit("Hugging Face dataset server rate-limited download; retry later")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--output", type=Path, default=ROOT / "testdata/mnsc-asr-part1")
    args = parser.parse_args()

    first = rows(0)
    total = first["num_rows_total"]
    if not 1 <= args.count <= total:
        raise SystemExit(f"--count must be 1–{total}")
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"refusing non-empty output directory: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    manifest = []
    width = total // args.count
    for start in (i * width for i in range(args.count)):
        candidates = first["rows"] if start == 0 else rows(start)["rows"]
        for row in candidates:
            index = row["row_idx"]
            item = row["row"]
            audio = args.output / f"{index:04d}.wav"
            with urlopen(item["context"][0]["src"], timeout=120) as response, audio.open("wb") as output:
                shutil.copyfileobj(response, output)
            with wave.open(str(audio)) as wav:
                duration = wav.getnframes() / wav.getframerate()
            if not 5 <= duration <= 30:
                audio.unlink()
                continue
            manifest.append({
                "dataset": DATASET,
                "config": CONFIG,
                "split": SPLIT,
                "row": index,
                "audio": audio.name,
                "seconds": round(duration, 3),
                "transcript": item["answer"],
            })
            print(f"{audio.name}: {duration:.1f}s")
            break
        else:
            raise SystemExit(f"no 5–30s clip found after row {start}")

    (args.output / "manifest.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifest))


if __name__ == "__main__":
    main()
