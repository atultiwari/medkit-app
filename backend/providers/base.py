"""Abstract base classes and shared DTOs for medkit's provider layer.

Three provider kinds:
  - LLMProvider — chat + structured grading.
  - STTProvider — speech-to-text used by the local voice worker.
  - TTSProvider — text-to-speech used by the local voice worker.

Each implementation lives in `backend/providers/<name>_provider.py`. The
factory (`factory.py`) reads env vars, instantiates the right provider,
and returns it. Callers depend only on the abstract base, so swapping
providers requires no changes outside this directory.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional


@dataclass(frozen=True)
class ChatMessage:
    """Single chat turn in a request to the LLM."""

    role: str  # 'user' | 'assistant' | 'system'
    content: str


@dataclass(frozen=True)
class ChatDelta:
    """Streaming delta yielded by `LLMProvider.stream_chat`.

    `text` is the incremental token chunk; `done=True` marks the final
    sentinel after the last delta.
    """

    text: str = ""
    done: bool = False
    error: Optional[str] = None


@dataclass(frozen=True)
class DebriefRequest:
    """Input for `LLMProvider.grade_attending`.

    Mirrors the `DebriefRequest` JSON the frontend sends when an encounter
    ends. Kept as a dataclass so providers can pass it to their own
    serializer without import-cycling on the Pydantic models in server.py.
    """

    case_id: str
    case_summary: dict[str, Any]
    rubric: dict[str, Any]
    registry_slice: dict[str, Any]
    encounter_log: dict[str, Any]


@dataclass(frozen=True)
class CaseEvaluation:
    """Output of `LLMProvider.grade_attending`. Shape mirrors the
    `render_case_evaluation` Zod schema in `src/agents/customTools.ts`."""

    case_id: str
    global_rating: str
    domain_scores: dict[str, dict[str, Any]]
    criteria: list[dict[str, Any]]
    highlights: list[str]
    improvements: list[str]
    narrative: str
    safety_breach: Optional[dict[str, Any]] = None


class LLMProvider(ABC):
    """Conversational + structured-output LLM."""

    name: str = "abstract"

    @property
    @abstractmethod
    def supports_tools(self) -> bool:
        """True iff the underlying model exposes OpenAI-style tool calling.

        Providers that return False fall back to JSON-schema prompting in
        their `grade_attending` implementation.
        """

    @abstractmethod
    async def stream_chat(
        self,
        messages: list[ChatMessage],
        system: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> AsyncIterator[ChatDelta]:
        """Stream a chat response. Yields `ChatDelta` chunks; the final
        delta has `done=True`. On error, yields a single delta with
        `error` set and `done=True`."""

    @abstractmethod
    async def grade_attending(self, request: DebriefRequest) -> CaseEvaluation:
        """Run the attending debrief grading pass and return a structured
        `CaseEvaluation`. Implementations must NEVER fabricate guideline
        refs that are not in `request.registry_slice` — drop the
        criterion instead.
        """


class STTProvider(ABC):
    """Speech-to-text. Used by the local voice worker."""

    name: str = "abstract"

    @abstractmethod
    async def transcribe(self, audio_bytes: bytes, sample_rate: int) -> str:
        """Transcribe a single audio buffer (16-bit PCM mono) and return
        the recognized text. For streaming providers, callers chunk the
        audio at the VAD boundary and call this per-chunk."""


class TTSProvider(ABC):
    """Text-to-speech. Used by the local voice worker."""

    name: str = "abstract"

    @abstractmethod
    async def synthesize(
        self,
        text: str,
        voice_id: str,
        sample_rate: int = 22050,
    ) -> AsyncIterator[bytes]:
        """Synthesize `text` and yield 16-bit PCM mono frames at the
        requested sample rate. Streaming providers yield as data arrives;
        non-streaming providers yield once at the end."""


@dataclass(frozen=True)
class ProviderChain:
    """Ordered fallback chain. The factory builds one of these from env
    and exposes it via `get_chain()`. Callers iterate `chain.providers`
    and use the first one whose health check passes."""

    providers: tuple[LLMProvider, ...] = field(default_factory=tuple)

    def __iter__(self):
        return iter(self.providers)

    @property
    def primary(self) -> LLMProvider:
        if not self.providers:
            raise RuntimeError("Provider chain is empty")
        return self.providers[0]
