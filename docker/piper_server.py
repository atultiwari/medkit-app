"""Minimal HTTP wrapper around Piper TTS.

Single endpoint:
    POST /synthesize     body={"text": "...", "voice": "en_US-amy-medium"}
    response: raw 16-bit PCM mono at the voice's native sample rate
              (typically 22.05 kHz for *-medium models)
"""

from __future__ import annotations

import io
import logging
import os
import wave
from threading import Lock
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

logger = logging.getLogger("medkit.piper-server")
logger.setLevel(logging.INFO)

VOICE_DIR = os.environ.get("PIPER_VOICE_DIR", "/models")
DEFAULT_VOICE = os.environ.get("PIPER_DEFAULT_VOICE", "en_US-ryan-medium")

app = FastAPI(title="medkit-piper", version="0.1.0")

_voice_cache: dict = {}
_lock = Lock()


def _load_voice(name: str):
    with _lock:
        if name in _voice_cache:
            return _voice_cache[name]
        try:
            from piper.voice import PiperVoice  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise HTTPException(
                status_code=500,
                detail=(
                    "piper-tts package missing. The container build should "
                    "install it; rebuild the image."
                ),
            ) from e
        model_path = os.path.join(VOICE_DIR, f"{name}.onnx")
        config_path = os.path.join(VOICE_DIR, f"{name}.onnx.json")
        if not os.path.exists(model_path):
            raise HTTPException(
                status_code=404,
                detail=f"voice not found: {name} (expected at {model_path})",
            )
        logger.info("loading piper voice %s", name)
        voice = PiperVoice.load(model_path, config_path=config_path)
        _voice_cache[name] = voice
        return voice


class SynthesizeRequest(BaseModel):
    text: str
    voice: Optional[str] = None


@app.get("/health")
def health() -> dict:
    voices: list[str] = []
    if os.path.isdir(VOICE_DIR):
        for entry in os.listdir(VOICE_DIR):
            if entry.endswith(".onnx"):
                voices.append(entry[:-5])
    return {
        "ok": True,
        "voice_dir": VOICE_DIR,
        "default_voice": DEFAULT_VOICE,
        "available": voices,
        "loaded": list(_voice_cache.keys()),
    }


@app.post("/synthesize")
def synthesize(req: SynthesizeRequest) -> Response:
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text is required")
    voice_name = (req.voice or DEFAULT_VOICE).strip()
    voice = _load_voice(voice_name)

    # Piper writes raw int16 PCM to a wave handle; we capture into memory.
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(voice.config.sample_rate)
        try:
            voice.synthesize(req.text, wf)
        except Exception as e:
            logger.exception("synthesis failed")
            raise HTTPException(status_code=500, detail=f"synthesis failed: {e}")
    # Strip the wav header to return raw PCM as the provider expects.
    buf.seek(0)
    with wave.open(buf, "rb") as wr:
        frames = wr.readframes(wr.getnframes())
    return Response(content=frames, media_type="audio/raw")
