"""Piper TTS provider — default for offline mode.

Two run modes (mirroring `WhisperProvider`):
  1. Sidecar — `PIPER_URL` is set; we POST {text, voice} and read PCM
     bytes from the response. Preferred for the Dockerised path.
  2. In-process — `piper` CLI on PATH; we shell out per request.

Piper voices are picked by gender + age:
  - en_US-ryan-medium    (adult M)
  - en_US-amy-medium     (adult F)
  - en_US-joe-medium     (parent M)
  - en_US-lessac-medium  (parent F)
  - en_US-kathleen-low   (child)

Override the mapping via `PIPER_VOICE_MAP` (JSON) if desired.

Env:
  PIPER_URL          — sidecar HTTP endpoint (preferred). e.g. http://piper:5002
  PIPER_VOICE_DIR    — local voices directory (in-process mode)
  PIPER_DEFAULT_VOICE — fallback voice id when caller doesn't pass one
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shlex
from typing import AsyncIterator, Optional

import httpx

from backend.providers.base import TTSProvider

_log = logging.getLogger("medkit.providers.piper")


_DEFAULT_VOICE_MAP = {
    "adult_male": "en_US-ryan-medium",
    "adult_female": "en_US-amy-medium",
    "parent_male": "en_US-joe-medium",
    "parent_female": "en_US-lessac-medium",
    "child": "en_US-kathleen-low",
}


class PiperProvider(TTSProvider):
    name = "piper"

    def __init__(self) -> None:
        self.sidecar_url = os.environ.get("PIPER_URL", "").strip().rstrip("/")
        self.voice_dir = os.environ.get("PIPER_VOICE_DIR", "/models/piper")
        self.default_voice = os.environ.get(
            "PIPER_DEFAULT_VOICE", _DEFAULT_VOICE_MAP["adult_male"]
        )
        try:
            map_raw = os.environ.get("PIPER_VOICE_MAP", "")
            self.voice_map = (
                {**_DEFAULT_VOICE_MAP, **json.loads(map_raw)}
                if map_raw
                else dict(_DEFAULT_VOICE_MAP)
            )
        except json.JSONDecodeError:
            _log.warning("PIPER_VOICE_MAP is not valid JSON; using defaults")
            self.voice_map = dict(_DEFAULT_VOICE_MAP)
        self.timeout = float(os.environ.get("PIPER_TIMEOUT_SEC", "30"))
        self._client: Optional[httpx.AsyncClient] = None

    def _http_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout, connect=10.0),
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()

    async def synthesize(
        self,
        text: str,
        voice_id: str,
        sample_rate: int = 22050,
    ) -> AsyncIterator[bytes]:
        resolved = self.voice_map.get(voice_id, voice_id) or self.default_voice
        if self.sidecar_url:
            async for chunk in self._synthesize_sidecar(text, resolved):
                yield chunk
            return
        chunk = await self._synthesize_local(text, resolved)
        yield chunk

    async def _synthesize_sidecar(
        self, text: str, voice_id: str
    ) -> AsyncIterator[bytes]:
        client = self._http_client()
        url = f"{self.sidecar_url}/synthesize"
        async with client.stream(
            "POST",
            url,
            json={"text": text, "voice": voice_id},
        ) as resp:
            if resp.status_code != 200:
                body = (await resp.aread()).decode("utf-8", errors="replace")
                raise RuntimeError(f"piper sidecar {resp.status_code}: {body}")
            async for chunk in resp.aiter_bytes(chunk_size=4096):
                if chunk:
                    yield chunk

    async def _synthesize_local(self, text: str, voice_id: str) -> bytes:
        voice_path = os.path.join(self.voice_dir, f"{voice_id}.onnx")
        cmd = ["piper", "--model", voice_path, "--output_raw"]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, err = await proc.communicate(input=text.encode("utf-8"))
        if proc.returncode != 0:
            raise RuntimeError(
                f"piper exited {proc.returncode}: {err.decode('utf-8', errors='replace')}"
            )
        _log.debug("piper local synth: cmd=%s bytes=%d", shlex.join(cmd), len(out))
        return out
