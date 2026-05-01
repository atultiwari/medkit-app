"""faster-whisper STT provider — default for offline mode.

Two run modes:
  1. In-process — `faster_whisper` package is installed in the backend
     venv, model loaded on first use. Works for synchronous backends.
  2. Sidecar — `WHISPER_URL` is set; we POST WAV bytes and read the
     transcript from a small HTTP server (see `docker/whisper.Dockerfile`).
     Preferred for the Dockerised path so the model isn't reloaded per
     process.

Env:
  WHISPER_URL    — sidecar HTTP endpoint (preferred). e.g. http://whisper:5001
  WHISPER_MODEL  — small.en | medium.en | large-v3 (default: small.en)
  WHISPER_DEVICE — cpu | cuda | auto (default: cpu)
"""

from __future__ import annotations

import io
import logging
import os
import wave
from typing import Optional

import httpx

from backend.providers.base import STTProvider

_log = logging.getLogger("medkit.providers.whisper")


class WhisperProvider(STTProvider):
    name = "whisper"

    def __init__(self) -> None:
        self.sidecar_url = os.environ.get("WHISPER_URL", "").strip().rstrip("/")
        self.model_size = os.environ.get("WHISPER_MODEL", "small.en")
        self.device = os.environ.get("WHISPER_DEVICE", "cpu")
        self.timeout = float(os.environ.get("WHISPER_TIMEOUT_SEC", "60"))
        self._client: Optional[httpx.AsyncClient] = None
        self._local_model = None  # lazy-initialised

    def _http_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout, connect=10.0),
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()

    async def transcribe(self, audio_bytes: bytes, sample_rate: int) -> str:
        if self.sidecar_url:
            return await self._transcribe_sidecar(audio_bytes, sample_rate)
        return self._transcribe_local(audio_bytes, sample_rate)

    async def _transcribe_sidecar(self, audio: bytes, sample_rate: int) -> str:
        wav = _pcm_to_wav(audio, sample_rate)
        client = self._http_client()
        resp = await client.post(
            f"{self.sidecar_url}/transcribe",
            content=wav,
            headers={"Content-Type": "audio/wav"},
            params={"model": self.model_size, "language": "en"},
        )
        if resp.status_code != 200:
            raise RuntimeError(f"whisper sidecar {resp.status_code}: {resp.text}")
        body = resp.json()
        return (body.get("text") or "").strip()

    def _transcribe_local(self, audio: bytes, sample_rate: int) -> str:
        if self._local_model is None:
            try:
                from faster_whisper import WhisperModel  # type: ignore
            except ImportError as e:  # pragma: no cover
                raise RuntimeError(
                    "faster_whisper not installed and WHISPER_URL is unset. "
                    "Run `pip install faster-whisper` or point WHISPER_URL "
                    "at the sidecar."
                ) from e
            _log.info("loading faster-whisper model %s on %s", self.model_size, self.device)
            self._local_model = WhisperModel(self.model_size, device=self.device)
        wav = _pcm_to_wav(audio, sample_rate)
        segments, _ = self._local_model.transcribe(
            io.BytesIO(wav), language="en", vad_filter=True
        )
        return "".join(seg.text for seg in segments).strip()


def _pcm_to_wav(pcm: bytes, sample_rate: int) -> bytes:
    """Wrap raw 16-bit PCM mono in a WAV header so HTTP sidecars can
    parse it without negotiating a sample rate out-of-band."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()
