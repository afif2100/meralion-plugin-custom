"""OpenAI-style transcription endpoint for MERaLiON chat models."""

import base64
import mimetypes
from http import HTTPStatus
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.speech_to_text.base.utils import read_upload_with_limit

router = APIRouter()
_ALLOWED_SUFFIXES = {".flac", ".m4a", ".mp3", ".mp4", ".mpeg", ".mpga", ".ogg", ".wav", ".webm"}
_PROMPT = "Instruction: Please transcribe this speech.\nFollow the text instruction based on the following audio: <SpeechHere>"


@router.post("/v1/audio/transcriptions")
async def create_transcription(
    raw_request: Request,
    file: Annotated[UploadFile, File()],
    model: Annotated[str | None, Form()] = None,
    language: Annotated[str | None, Form()] = None,
    prompt: Annotated[str, Form()] = "",
    response_format: Annotated[str, Form()] = "json",
    temperature: Annotated[float, Form()] = 0,
):
    """Transcribe one audio file through MERaLiON's chat interface."""
    suffix = (file.filename or "").lower().rsplit(".", 1)
    suffix = f".{suffix[-1]}" if len(suffix) == 2 else ""
    if suffix not in _ALLOWED_SUFFIXES:
        raise HTTPException(
            HTTPStatus.BAD_REQUEST,
            f"Unsupported audio type {suffix or '(none)'}.",
        )
    if len(prompt) > 4096:
        raise HTTPException(HTTPStatus.BAD_REQUEST, "prompt exceeds 4096 characters.")
    if response_format not in {"json", "text"}:
        raise HTTPException(
            HTTPStatus.BAD_REQUEST,
            "response_format must be 'json' or 'text'; timestamps are unavailable.",
        )

    audio = await read_upload_with_limit(file)
    if not audio:
        raise HTTPException(HTTPStatus.BAD_REQUEST, "Audio file is empty.")

    mime = file.content_type or mimetypes.guess_type(file.filename or "")[0]
    mime = mime if mime and mime != "application/octet-stream" else "audio/" + suffix[1:]
    instruction = _PROMPT
    if language:
        instruction += f"\nSpoken language: {language}."
    if prompt:
        instruction += f"\nContext: {prompt}"

    request = ChatCompletionRequest(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {
                        "type": "audio_url",
                        "audio_url": {
                            "url": f"data:{mime};base64,{base64.b64encode(audio).decode()}"
                        },
                    },
                ],
            }
        ],
        max_completion_tokens=448,
        temperature=temperature,
    )
    handler = raw_request.app.state.openai_serving_chat
    result = await handler.create_chat_completion(request, raw_request)

    if isinstance(result, ErrorResponse):
        return JSONResponse(result.model_dump(), status_code=result.error.code)
    if not isinstance(result, ChatCompletionResponse):
        raise HTTPException(HTTPStatus.NOT_IMPLEMENTED, "Streaming transcriptions are unsupported.")

    text = result.choices[0].message.content or ""
    if response_format == "text":
        return PlainTextResponse(text)
    return JSONResponse({"text": text})


class MERaLiONWhisperTranscriptionPlugin:
    """Attach the opt-in OpenAI transcription compatibility route."""

    name = "meralion_whisper_transcription"
    required_tasks = ("generate",)

    def attach_router(self, app):
        app.include_router(router)

    async def init_state(self, engine_client, state, args):
        return None
