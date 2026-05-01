"""Provider factory.

Reads environment variables and returns the active provider for each
capability. Offline (Ollama + faster-whisper + Piper) is the default.

Env vars:
  LLM_PROVIDER       ollama | gemini | claude     (default: ollama)
  STT_PROVIDER       whisper | deepgram           (default: whisper)
  TTS_PROVIDER       piper | cartesia             (default: piper)
  LLM_FALLBACK_CHAIN comma-separated list, e.g. "ollama,gemini,claude"
                     (default: derived from LLM_PROVIDER)
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from backend.providers.base import (
    LLMProvider,
    ProviderChain,
    STTProvider,
    TTSProvider,
)

_log = logging.getLogger("medkit.providers")
_log.setLevel(logging.INFO)


_DEFAULT_LLM = "ollama"
_DEFAULT_STT = "whisper"
_DEFAULT_TTS = "piper"


# Singletons — providers are stateless per-request but expensive to
# construct (HTTP clients, model handles). Cache by name.
_llm_cache: dict[str, LLMProvider] = {}
_stt_cache: dict[str, STTProvider] = {}
_tts_cache: dict[str, TTSProvider] = {}


def _build_llm(name: str) -> LLMProvider:
    name = name.lower().strip()
    if name == "ollama":
        from backend.providers.ollama_provider import OllamaProvider

        return OllamaProvider()
    if name == "gemini":
        from backend.providers.gemini_provider import GeminiProvider

        return GeminiProvider()
    if name == "claude":
        from backend.providers.claude_provider import ClaudeProvider

        return ClaudeProvider()
    raise ValueError(f"Unknown LLM_PROVIDER: {name!r}")


def _build_stt(name: str) -> STTProvider:
    name = name.lower().strip()
    if name == "whisper":
        from backend.providers.stt.whisper_provider import WhisperProvider

        return WhisperProvider()
    if name == "deepgram":
        from backend.providers.stt.deepgram_provider import DeepgramProvider

        return DeepgramProvider()
    raise ValueError(f"Unknown STT_PROVIDER: {name!r}")


def _build_tts(name: str) -> TTSProvider:
    name = name.lower().strip()
    if name == "piper":
        from backend.providers.tts.piper_provider import PiperProvider

        return PiperProvider()
    if name == "cartesia":
        from backend.providers.tts.cartesia_provider import CartesiaProvider

        return CartesiaProvider()
    raise ValueError(f"Unknown TTS_PROVIDER: {name!r}")


def get_llm_provider(name: Optional[str] = None) -> LLMProvider:
    """Return the configured LLM provider. Cached."""
    chosen = (name or os.environ.get("LLM_PROVIDER") or _DEFAULT_LLM).lower().strip()
    if chosen not in _llm_cache:
        _llm_cache[chosen] = _build_llm(chosen)
        _log.info("llm provider initialised: %s", chosen)
    return _llm_cache[chosen]


def get_stt_provider(name: Optional[str] = None) -> STTProvider:
    chosen = (name or os.environ.get("STT_PROVIDER") or _DEFAULT_STT).lower().strip()
    if chosen not in _stt_cache:
        _stt_cache[chosen] = _build_stt(chosen)
        _log.info("stt provider initialised: %s", chosen)
    return _stt_cache[chosen]


def get_tts_provider(name: Optional[str] = None) -> TTSProvider:
    chosen = (name or os.environ.get("TTS_PROVIDER") or _DEFAULT_TTS).lower().strip()
    if chosen not in _tts_cache:
        _tts_cache[chosen] = _build_tts(chosen)
        _log.info("tts provider initialised: %s", chosen)
    return _tts_cache[chosen]


def get_chain() -> ProviderChain:
    """Build the LLM fallback chain from env. The primary is
    `LLM_PROVIDER`; subsequent entries come from `LLM_FALLBACK_CHAIN`.
    Duplicates are removed in order.
    """
    primary = (os.environ.get("LLM_PROVIDER") or _DEFAULT_LLM).lower().strip()
    raw_chain = os.environ.get("LLM_FALLBACK_CHAIN", "").strip()
    extras = [p.strip().lower() for p in raw_chain.split(",") if p.strip()]
    ordered: list[str] = []
    for candidate in [primary, *extras]:
        if candidate and candidate not in ordered:
            ordered.append(candidate)
    providers: list[LLMProvider] = []
    for name in ordered:
        try:
            providers.append(get_llm_provider(name))
        except Exception as e:  # pragma: no cover
            _log.warning("provider %r unavailable, skipping: %s", name, e)
    if not providers:
        # Last-ditch: surface a clear error rather than a cryptic empty chain.
        raise RuntimeError(
            "No LLM providers could be initialised. Check LLM_PROVIDER and "
            "LLM_FALLBACK_CHAIN env vars."
        )
    return ProviderChain(providers=tuple(providers))
