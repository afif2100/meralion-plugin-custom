#!/usr/bin/env python3
"""Score MERaLiON ASR output against a local manifest of transcripts."""

import argparse
import base64
import json
import os
import re
import unicodedata
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent.parent
PROMPT = "Instruction: Please transcribe this speech.\nFollow the text instruction based on the following audio: <SpeechHere>"
MIME = {".flac": "audio/flac", ".ogg": "audio/ogg", ".mp3": "audio/mpeg", ".wav": "audio/wav"}


def tokens(text: str) -> list[str]:
    return re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", text).casefold())


def word_errors(reference: list[str], prediction: list[str]) -> int:
    previous = list(range(len(prediction) + 1))
    for row, expected in enumerate(reference, 1):
        current = [row]
        for column, actual in enumerate(prediction, 1):
            current.append(min(current[-1] + 1, previous[column] + 1, previous[column - 1] + (expected != actual)))
        previous = current
    return previous[-1]


def transcribe(audio: Path, api_url: str, model: str) -> str:
    encoded = base64.b64encode(audio.read_bytes()).decode()
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": PROMPT},
            {"type": "audio_url", "audio_url": {"url": f"data:{MIME.get(audio.suffix.lower(), 'application/octet-stream')};base64,{encoded}"}},
        ]}],
        "max_completion_tokens": 256,
        "temperature": 0,
    }
    request = Request(f"{api_url}/v1/chat/completions", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=300) as response:
        return json.load(response)["choices"][0]["message"]["content"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "testdata/mnsc-asr-part1/manifest.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "testdata/mnsc-asr-part1/results.jsonl")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    if args.output.exists():
        raise SystemExit(f"refusing to overwrite results: {args.output}")
    entries = [json.loads(line) for line in args.manifest.read_text().splitlines()]
    if args.limit:
        entries = entries[:args.limit]
    if not entries:
        raise SystemExit("manifest has no entries")

    api_url = os.environ.get("MERALION_API_URL", "http://127.0.0.1:8000")
    model = os.environ.get("MERALION_MODEL", str(ROOT / "models/MERaLiON/MERaLiON-2-10B-ASR"))
    results = []
    for number, entry in enumerate(entries, 1):
        reference = tokens(entry["transcript"])
        prediction = transcribe(args.manifest.parent / entry["audio"], api_url, model)
        actual = tokens(prediction)
        errors = word_errors(reference, actual)
        results.append({**entry, "prediction": prediction, "word_errors": errors, "reference_words": len(reference), "wer": errors / len(reference)})
        print(f"{number}/{len(entries)} {entry['audio']}: {errors / len(reference):.1%} WER")

    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in results))
    errors = sum(row["word_errors"] for row in results)
    words = sum(row["reference_words"] for row in results)
    print(f"WER: {errors / words:.1%} ({errors}/{words} words)")


if __name__ == "__main__":
    assert word_errors(tokens("one two"), tokens("one three")) == 1
    main()
