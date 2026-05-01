"""Deepgram STT provider — optional cloud fallback.

Used when `STT_PROVIDER=deepgram`. The LiveKit voice worker has a native
Deepgram plugin (`livekit.plugins.deepgram`) that does streaming directly
through LiveKit; this provider exists for one-shot transcription via the
provider abstraction (debug tools, tests).
"""

from __future__ import annotations

import logging
import os

import httpx

from backend.providers.base import STTProvider

_log = logging.getLogger("medkit.providers.deepgram")


class DeepgramProvider(STTProvider):
    name = "deepgram"

    def __init__(self) -> None:
        self.api_key = os.environ.get("DEEPGRAM_API_KEY", "").strip()
        if not self.api_key:
            raise RuntimeError(
                "DEEPGRAM_API_KEY is not set. Cannot initialise Deepgram provider."
            )
        self.model = os.environ.get("DEEPGRAM_MODEL", "nova-3")
        self.timeout = float(os.environ.get("DEEPGRAM_TIMEOUT_SEC", "30"))
        self._client = httpx.AsyncClient(
            base_url="https://api.deepgram.com",
            timeout=httpx.Timeout(self.timeout, connect=10.0),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def transcribe(self, audio_bytes: bytes, sample_rate: int) -> str:
        params = {
            "model": self.model,
            "language": "en",
            "encoding": "linear16",
            "sample_rate": str(sample_rate),
            "channels": "1",
        }
        headers = {
            "Authorization": f"Token {self.api_key}",
            "Content-Type": "audio/raw",
        }
        resp = await self._client.post(
            "/v1/listen", params=params, headers=headers, content=audio_bytes
        )
        if resp.status_code != 200:
            raise RuntimeError(f"deepgram {resp.status_code}: {resp.text}")
        data = resp.json()
        try:
            return (
                data["results"]["channels"][0]["alternatives"][0]["transcript"]
                or ""
            ).strip()
        except (KeyError, IndexError, TypeError):
            return ""
