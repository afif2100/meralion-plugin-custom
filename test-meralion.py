#!/usr/bin/env python3
import base64
import json
import os
import sys
from pathlib import Path
from urllib.request import Request, urlopen

path = Path(sys.argv[1] if len(sys.argv) > 1 else "testdata/librispeech-sample.flac")
audio = base64.b64encode(path.read_bytes()).decode()
mime = {".flac": "audio/flac", ".ogg": "audio/ogg", ".mp3": "audio/mpeg", ".wav": "audio/wav"}.get(path.suffix.lower(), "application/octet-stream")
payload = {
    "model": os.environ.get("MERALION_MODEL", "models/MERaLiON/MERaLiON-2-10B-ASR"),
    "messages": [{"role": "user", "content": [
        {"type": "text", "text": "Instruction: Please transcribe this speech.\nFollow the text instruction based on the following audio: <SpeechHere>"},
        {"type": "audio_url", "audio_url": {"url": f"data:{mime};base64,{audio}"}},
    ]}],
    "max_completion_tokens": 256,
    "temperature": 0,
}
request = Request(f"{os.environ.get('MERALION_API_URL', 'http://127.0.0.1:8000')}/v1/chat/completions", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
with urlopen(request, timeout=300) as response:
    print(json.load(response)["choices"][0]["message"]["content"])
