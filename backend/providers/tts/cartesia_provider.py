"""Cartesia TTS provider — optional cloud fallback.

The LiveKit voice worker has a native Cartesia plugin
(`livekit.plugins.cartesia`) that streams directly through LiveKit; this
provider exists for one-shot synthesis via the provider abstraction
(debug tools, tests).
"""

from __future__ import annotations

import logging
import os
from typing import AsyncIterator

import httpx

from backend.providers.base import TTSProvider

_log = logging.getLogger("medkit.providers.cartesia")


class CartesiaProvider(TTSProvider):
    name = "cartesia"

    def __init__(self) -> None:
        self.api_key = os.environ.get("CARTESIA_API_KEY", "").strip()
        if not self.api_key:
            raise RuntimeError(
                "CARTESIA_API_KEY is not set. Cannot initialise Cartesia provider."
            )
        self.model = os.environ.get("CARTESIA_MODEL", "sonic-2")
        self.timeout = float(os.environ.get("CARTESIA_TIMEOUT_SEC", "30"))
        self.api_version = os.environ.get("CARTESIA_API_VERSION", "2024-06-10")
        self._client = httpx.AsyncClient(
            base_url="https://api.cartesia.ai",
            timeout=httpx.Timeout(self.timeout, connect=10.0),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def synthesize(
        self,
        text: str,
        voice_id: str,
        sample_rate: int = 22050,
    ) -> AsyncIterator[bytes]:
        headers = {
            "X-API-Key": self.api_key,
            "Cartesia-Version": self.api_version,
            "Content-Type": "application/json",
        }
        body = {
            "model_id": self.model,
            "transcript": text,
            "voice": {"mode": "id", "id": voice_id},
            "output_format": {
                "container": "raw",
                "encoding": "pcm_s16le",
                "sample_rate": sample_rate,
            },
        }
        async with self._client.stream(
            "POST", "/tts/bytes", headers=headers, json=body
        ) as resp:
            if resp.status_code != 200:
                err = (await resp.aread()).decode("utf-8", errors="replace")
                raise RuntimeError(f"cartesia {resp.status_code}: {err}")
            async for chunk in resp.aiter_bytes(chunk_size=4096):
                if chunk:
                    yield chunk
