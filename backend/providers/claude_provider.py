"""Claude LLM provider — last-priority cloud fallback.

Wraps the Anthropic SDK so the rest of the system can speak provider-
neutral DTOs. The existing Managed Agents endpoints in `server.py` keep
their direct SDK usage (the agent objects are stateful), but
`stream_chat` and `grade_attending` route through this provider when
LLM_PROVIDER=claude.

Defaults:
  CLAUDE_MODEL_PATIENT   = claude-haiku-4-5
  CLAUDE_MODEL_ATTENDING = claude-opus-4-7
  ANTHROPIC_API_KEY      (required)
"""

from __future__ import annotations

import json
import logging
import os
from typing import AsyncIterator, Optional

from backend.providers.base import (
    CaseEvaluation,
    ChatDelta,
    ChatMessage,
    DebriefRequest,
    LLMProvider,
)
from backend.providers.ollama_provider import (
    _ATTENDING_SYSTEM,
    _collect_valid_refs,
)

_log = logging.getLogger("medkit.providers.claude")


class ClaudeProvider(LLMProvider):
    name = "claude"

    def __init__(self) -> None:
        try:
            from anthropic import AsyncAnthropic  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "anthropic package not installed. "
                "Run `pip install 'anthropic>=0.88.0'` in the backend venv."
            ) from e
        api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set.")
        self._AsyncAnthropic = AsyncAnthropic
        self._client = AsyncAnthropic(api_key=api_key)
        self.patient_model = os.environ.get(
            "CLAUDE_MODEL_PATIENT", "claude-haiku-4-5"
        )
        self.attending_model = os.environ.get(
            "CLAUDE_MODEL_ATTENDING", "claude-opus-4-7"
        )

    @property
    def supports_tools(self) -> bool:
        return True

    async def stream_chat(
        self,
        messages: list[ChatMessage],
        system: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> AsyncIterator[ChatDelta]:
        try:
            kwargs = {
                "model": self.patient_model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [
                    {"role": m.role, "content": m.content}
                    for m in messages
                    if m.role in ("user", "assistant")
                ],
            }
            if system:
                kwargs["system"] = [
                    {
                        "type": "text",
                        "text": system,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            async with self._client.messages.stream(**kwargs) as stream:  # type: ignore[attr-defined]
                async for event in stream:
                    etype = getattr(event, "type", None)
                    if etype != "content_block_delta":
                        continue
                    delta = getattr(event, "delta", None)
                    if getattr(delta, "type", None) != "text_delta":
                        continue
                    text = getattr(delta, "text", "")
                    if text:
                        yield ChatDelta(text=text)
            yield ChatDelta(done=True)
        except Exception as e:
            _log.exception("claude stream_chat failed")
            yield ChatDelta(error=str(e), done=True)

    async def grade_attending(self, request: DebriefRequest) -> CaseEvaluation:
        user_payload = {
            "case_id": request.case_id,
            "case_summary": request.case_summary,
            "rubric": request.rubric,
            "registry_slice": request.registry_slice,
            "encounter_log": request.encounter_log,
        }
        msg = await self._client.messages.create(  # type: ignore[attr-defined]
            model=self.attending_model,
            max_tokens=4096,
            temperature=0.2,
            system=[
                {
                    "type": "text",
                    "text": _ATTENDING_SYSTEM,
                }
            ],
            messages=[
                {
                    "role": "user",
                    "content": json.dumps(user_payload, ensure_ascii=False),
                }
            ],
        )
        text_blocks: list[str] = []
        for b in getattr(msg, "content", []) or []:
            if getattr(b, "type", None) == "text":
                t = getattr(b, "text", "")
                if isinstance(t, str):
                    text_blocks.append(t)
        text = "".join(text_blocks).strip()
        if not text:
            raise RuntimeError("claude returned empty content for attending grading")
        return _parse_evaluation_shared(text, request)


def _parse_evaluation_shared(text: str, request: DebriefRequest) -> CaseEvaluation:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"claude attending output is not valid JSON: {e}") from e
    valid_refs = _collect_valid_refs(request.registry_slice)
    criteria = []
    for raw in data.get("criteria", []) or []:
        ref = raw.get("guideline_ref")
        if ref is not None and ref not in valid_refs:
            raw = {**raw, "guideline_ref": None}
        criteria.append(raw)
    return CaseEvaluation(
        case_id=str(data.get("case_id", request.case_id)),
        global_rating=str(data.get("global_rating", "borderline")),
        domain_scores=data.get("domain_scores", {}),
        criteria=criteria,
        highlights=list(data.get("highlights", [])),
        improvements=list(data.get("improvements", [])),
        narrative=str(data.get("narrative", "")),
        safety_breach=data.get("safety_breach"),
    )
