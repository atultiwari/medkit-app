"""Minimal HTTP wrapper around faster-whisper.

Single endpoint:
    POST /transcribe        body=audio/wav (16-bit PCM)
                            params: model=small.en, language=en
    response: {"text": "..."}

The model is cached after first load; subsequent requests share the same
WhisperModel instance.
"""

from __future__ import annotations

import io
import logging
import os
from threading import Lock
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from faster_whisper import WhisperModel

logger = logging.getLogger("medkit.whisper-server")
logger.setLevel(logging.INFO)

DEFAULT_MODEL = os.environ.get("WHISPER_MODEL", "small.en")
DEFAULT_DEVICE = os.environ.get("WHISPER_DEVICE", "cpu")
DEFAULT_COMPUTE = os.environ.get(
    "WHISPER_COMPUTE_TYPE",
    "int8" if DEFAULT_DEVICE == "cpu" else "float16",
)

app = FastAPI(title="medkit-whisper", version="0.1.0")

_model_cache: dict[str, WhisperModel] = {}
_lock = Lock()


def get_model(name: str) -> WhisperModel:
    with _lock:
        if name not in _model_cache:
            logger.info(
                "loading faster-whisper %s on %s (%s)",
                name, DEFAULT_DEVICE, DEFAULT_COMPUTE,
            )
            _model_cache[name] = WhisperModel(
                name, device=DEFAULT_DEVICE, compute_type=DEFAULT_COMPUTE
            )
        return _model_cache[name]


@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "default_model": DEFAULT_MODEL,
        "device": DEFAULT_DEVICE,
        "compute_type": DEFAULT_COMPUTE,
        "loaded": list(_model_cache.keys()),
    }


@app.post("/transcribe")
async def transcribe(
    request: Request,
    model: Optional[str] = Query(default=None),
    language: str = Query(default="en"),
) -> JSONResponse:
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="empty request body")
    chosen_model = (model or DEFAULT_MODEL).strip()
    try:
        whisper = get_model(chosen_model)
    except Exception as e:
        logger.exception("failed to load model %s", chosen_model)
        raise HTTPException(status_code=500, detail=f"failed to load model: {e}")
    try:
        segments, _info = whisper.transcribe(
            io.BytesIO(body), language=language or None, vad_filter=True
        )
        text = "".join(seg.text for seg in segments).strip()
    except Exception as e:
        logger.exception("transcribe failed")
        raise HTTPException(status_code=500, detail=f"transcribe failed: {e}")
    return JSONResponse({"text": text, "model": chosen_model})
