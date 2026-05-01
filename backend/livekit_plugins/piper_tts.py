"""LiveKit TTS adapter that routes synthesis through `PiperProvider`.

Piper outputs 22.05 kHz 16-bit PCM mono. We chunk the response and emit
LiveKit `SynthesisedAudio` frames so the agent session can stream them
back to the browser as the speaker's audio track.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from livekit import rtc
from livekit.agents import tts

from backend.providers import get_tts_provider

_log = logging.getLogger("medkit.livekit_plugins.piper_tts")


_SAMPLE_RATE = 22050
_CHANNELS = 1


class PiperTTS(tts.TTS):
    def __init__(self, *, voice: str = "adult_male") -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=_SAMPLE_RATE,
            num_channels=_CHANNELS,
        )
        self._voice = voice
        self._provider = get_tts_provider("piper")

    def synthesize(self, text: str) -> tts.ChunkedStream:
        return _PiperChunkedStream(self, text, self._voice, self._provider)


class _PiperChunkedStream(tts.ChunkedStream):
    def __init__(
        self,
        owner: PiperTTS,
        text: str,
        voice: str,
        provider,
        *,
        conn_options: Optional[object] = None,
    ) -> None:
        super().__init__(tts=owner, input_text=text, conn_options=conn_options)
        self._voice = voice
        self._provider = provider

    async def _run(self):
        try:
            async for pcm in self._provider.synthesize(
                self.input_text, self._voice, sample_rate=_SAMPLE_RATE
            ):
                if not pcm:
                    continue
                # rtc.AudioFrame expects samples-per-channel count; PCM is
                # 2 bytes per sample (s16).
                samples_per_channel = len(pcm) // 2 // _CHANNELS
                if samples_per_channel <= 0:
                    continue
                frame = rtc.AudioFrame(
                    data=pcm,
                    sample_rate=_SAMPLE_RATE,
                    num_channels=_CHANNELS,
                    samples_per_channel=samples_per_channel,
                )
                self._event_ch.send_nowait(
                    tts.SynthesizedAudio(frame=frame, request_id=self._request_id)
                )
        except asyncio.CancelledError:
            raise
        except Exception as e:  # pragma: no cover
            _log.exception("piper synthesize failed: %s", e)
            raise
