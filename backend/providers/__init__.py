"""Provider abstraction for medkit.

The factory in `factory.py` selects an LLM, STT, or TTS implementation at
runtime based on environment variables. Offline (Ollama + faster-whisper +
Piper) is the default; Gemini and Claude are opt-in fallbacks.
"""

from backend.providers.base import (
    CaseEvaluation,
    ChatDelta,
    ChatMessage,
    DebriefRequest,
    LLMProvider,
    STTProvider,
    TTSProvider,
)
from backend.providers.factory import (
    get_llm_provider,
    get_stt_provider,
    get_tts_provider,
    get_chain,
)

__all__ = [
    "CaseEvaluation",
    "ChatDelta",
    "ChatMessage",
    "DebriefRequest",
    "LLMProvider",
    "STTProvider",
    "TTSProvider",
    "get_llm_provider",
    "get_stt_provider",
    "get_tts_provider",
    "get_chain",
]
