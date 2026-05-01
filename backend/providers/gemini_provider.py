"""Gemini LLM provider — optional cloud fallback.

Defaults:
  GEMINI_MODEL_PATIENT   = gemini-1.5-flash
  GEMINI_MODEL_ATTENDING = gemini-1.5-pro
  GEMINI_API_KEY         (required)

Uses the v1beta REST API directly (no SDK) so this module imports cleanly
even without `google-generativeai` installed in the venv.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, AsyncIterator, Optional

import httpx

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

_log = logging.getLogger("medkit.providers.gemini")


_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(self) -> None:
        self.api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not self.api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Cannot initialise Gemini provider."
            )
        self.patient_model = os.environ.get(
            "GEMINI_MODEL_PATIENT", "gemini-1.5-flash"
        )
        self.attending_model = os.environ.get(
            "GEMINI_MODEL_ATTENDING", "gemini-1.5-pro"
        )
        self.timeout = float(os.environ.get("GEMINI_TIMEOUT_SEC", "120"))
        self._client = httpx.AsyncClient(
            base_url=_BASE_URL,
            timeout=httpx.Timeout(self.timeout, connect=10.0),
        )

    @property
    def supports_tools(self) -> bool:
        # Gemini supports function calling; we still use JSON-mode for
        # the attending grader so the parser stays shared with Ollama.
        return True

    async def aclose(self) -> None:
        await self._client.aclose()

    async def stream_chat(
        self,
        messages: list[ChatMessage],
        system: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> AsyncIterator[ChatDelta]:
        body = _to_gemini_request(messages, system, max_tokens, temperature)
        url = f"/models/{self.patient_model}:streamGenerateContent"
        params = {"key": self.api_key, "alt": "sse"}
        try:
            async with self._client.stream(
                "POST", url, params=params, json=body
            ) as resp:
                if resp.status_code != 200:
                    text = (await resp.aread()).decode("utf-8", errors="replace")
                    yield ChatDelta(error=f"gemini {resp.status_code}: {text}", done=True)
                    return
                async for raw_line in resp.aiter_lines():
                    if not raw_line.startswith("data:"):
                        continue
                    payload_str = raw_line[5:].strip()
                    if not payload_str or payload_str == "[DONE]":
                        continue
                    try:
                        chunk = json.loads(payload_str)
                    except json.JSONDecodeError:
                        continue
                    for part_text in _extract_gemini_text(chunk):
                        if part_text:
                            yield ChatDelta(text=part_text)
                yield ChatDelta(done=True)
        except httpx.HTTPError as e:
            _log.exception("gemini stream_chat failed")
            yield ChatDelta(error=str(e), done=True)

    async def grade_attending(self, request: DebriefRequest) -> CaseEvaluation:
        user_payload = {
            "case_id": request.case_id,
            "case_summary": request.case_summary,
            "rubric": request.rubric,
            "registry_slice": request.registry_slice,
            "encounter_log": request.encounter_log,
        }
        body = {
            "systemInstruction": {"parts": [{"text": _ATTENDING_SYSTEM}]},
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": json.dumps(user_payload, ensure_ascii=False)}],
                }
            ],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 4096,
                "responseMimeType": "application/json",
            },
        }
        url = f"/models/{self.attending_model}:generateContent"
        resp = await self._client.post(
            url, params={"key": self.api_key}, json=body
        )
        if resp.status_code != 200:
            raise RuntimeError(f"gemini {resp.status_code}: {resp.text}")
        data = resp.json()
        text = "".join(_extract_gemini_text(data))
        if not text:
            raise RuntimeError("gemini returned empty content for attending grading")
        return _parse_evaluation_shared(text, request)


def _to_gemini_request(
    messages: list[ChatMessage],
    system: Optional[str],
    max_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    contents = []
    for m in messages:
        role = "user" if m.role == "user" else "model"
        contents.append({"role": role, "parts": [{"text": m.content}]})
    body: dict[str, Any] = {
        "contents": contents,
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        },
    }
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    return body


def _extract_gemini_text(payload: dict[str, Any]):
    candidates = payload.get("candidates") or []
    for c in candidates:
        content = c.get("content") or {}
        for part in content.get("parts") or []:
            txt = part.get("text")
            if isinstance(txt, str):
                yield txt


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
        raise RuntimeError(f"gemini attending output is not valid JSON: {e}") from e
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
