"""LiveKit STT adapter that routes audio through `WhisperProvider`.

LiveKit's `stt.STT` base class exposes `_recognize_impl(buffer)` for
non-streaming recognition. We collect the buffer's frames into a single
PCM blob and hand it to the provider. This is non-streaming — the
upstream VAD chunks audio at utterance boundaries before it reaches us,
so a per-utterance round-trip is the right granularity.

Streaming faster-whisper exists upstream but is finicky on CPU; we keep
this adapter simple and rely on VAD chunking for latency control.
"""

from __future__ import annotations

import logging
from typing import Optional

from livekit.agents import stt, utils
from livekit.agents.stt import STTCapabilities

from backend.providers import get_stt_provider

_log = logging.getLogger("medkit.livekit_plugins.whisper_stt")


class WhisperSTT(stt.STT):
    def __init__(self, *, language: str = "en") -> None:
        super().__init__(
            capabilities=STTCapabilities(streaming=False, interim_results=False),
        )
        self._language = language
        self._provider = get_stt_provider("whisper")

    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        language: Optional[str] = None,
        conn_options: Optional[object] = None,
    ) -> stt.SpeechEvent:
        # `buffer` is a `rtc.AudioFrame` or list of frames; LiveKit's
        # AudioBuffer abstraction handles both. Combine to a single frame
        # and grab the raw PCM data.
        frame = utils.merge_frames(buffer)
        pcm = bytes(frame.data)
        sample_rate = frame.sample_rate
        text = ""
        try:
            text = await self._provider.transcribe(pcm, sample_rate)
        except Exception as e:  # pragma: no cover — surfaced upstream
            _log.exception("whisper transcribe failed: %s", e)
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[
                stt.SpeechData(language=language or self._language, text=text)
            ],
        )
