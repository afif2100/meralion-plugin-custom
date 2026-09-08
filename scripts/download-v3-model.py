#!/usr/bin/env python3
"""Download MERaLiON-3-3B-ASR into this repository's ignored models directory."""

from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent.parent
MODEL = "MERaLiON/MERaLiON-3-3B-ASR"

snapshot_download(MODEL, local_dir=ROOT / "models/MERaLiON/MERaLiON-3-3B-ASR")
