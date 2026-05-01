"""Ollama LLM provider — default for offline mode.

Default model: `medgemma1.5:4b`. Override per-role via:
  OLLAMA_MODEL_PATIENT
  OLLAMA_MODEL_ATTENDING

The Gemma family does not provide reliable OpenAI-style tool calling on
small variants, so `supports_tools` returns False and `grade_attending`
uses JSON-schema prompting instead. The shared parser produces the same
`CaseEvaluation` shape Claude's tool-call path emits.
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

_log = logging.getLogger("medkit.providers.ollama")


_DEFAULT_BASE_URL = "http://127.0.0.1:11434"
_DEFAULT_MODEL = "medgemma1.5:4b"


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(self) -> None:
        self.base_url = os.environ.get("OLLAMA_BASE_URL", _DEFAULT_BASE_URL).rstrip("/")
        self.patient_model = os.environ.get("OLLAMA_MODEL_PATIENT", _DEFAULT_MODEL)
        self.attending_model = os.environ.get("OLLAMA_MODEL_ATTENDING", _DEFAULT_MODEL)
        self.timeout = float(os.environ.get("OLLAMA_TIMEOUT_SEC", "120"))
        # Long-lived client — Ollama keeps the connection alive and serves
        # multiple requests over it.
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(self.timeout, connect=10.0),
        )

    @property
    def supports_tools(self) -> bool:
        return False

    async def aclose(self) -> None:
        await self._client.aclose()

    async def stream_chat(
        self,
        messages: list[ChatMessage],
        system: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> AsyncIterator[ChatDelta]:
        payload: dict[str, Any] = {
            "model": self.patient_model,
            "messages": _to_ollama_messages(messages, system),
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        try:
            async with self._client.stream(
                "POST", "/api/chat", json=payload
            ) as resp:
                if resp.status_code != 200:
                    body = (await resp.aread()).decode("utf-8", errors="replace")
                    yield ChatDelta(error=f"ollama {resp.status_code}: {body}", done=True)
                    return
                async for raw_line in resp.aiter_lines():
                    if not raw_line:
                        continue
                    try:
                        chunk = json.loads(raw_line)
                    except json.JSONDecodeError:
                        continue
                    msg = chunk.get("message") or {}
                    content = msg.get("content") or ""
                    if content:
                        yield ChatDelta(text=content)
                    if chunk.get("done"):
                        yield ChatDelta(done=True)
                        return
                # If we exhaust the stream without seeing done=true (network
                # truncation), still emit a terminal sentinel so the caller
                # can finalise.
                yield ChatDelta(done=True)
        except httpx.HTTPError as e:
            _log.exception("ollama stream_chat failed")
            yield ChatDelta(error=str(e), done=True)

    async def grade_attending(self, request: DebriefRequest) -> CaseEvaluation:
        system, user = _build_attending_prompt(request)
        payload: dict[str, Any] = {
            "model": self.attending_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.2,
                "num_predict": 4096,
            },
        }
        resp = await self._client.post("/api/chat", json=payload)
        if resp.status_code != 200:
            raise RuntimeError(f"ollama {resp.status_code}: {resp.text}")
        body = resp.json()
        message = body.get("message") or {}
        text = (message.get("content") or "").strip()
        if not text:
            raise RuntimeError("ollama returned empty content for attending grading")
        return _parse_evaluation(text, request)


def _to_ollama_messages(
    messages: list[ChatMessage], system: Optional[str]
) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    if system:
        out.append({"role": "system", "content": system})
    for m in messages:
        out.append({"role": m.role, "content": m.content})
    return out


_ATTENDING_SYSTEM = (
    "You are the attending physician supervising a clinical training "
    "simulator. The trainee has just finished an encounter; your job is "
    "to grade it.\n\n"
    "RESPOND WITH STRICT JSON ONLY — no prose, no Markdown fences, no "
    "explanations outside the JSON object. The schema is:\n\n"
    "{\n"
    '  "case_id": string,\n'
    '  "global_rating": "clear-fail" | "borderline" | "satisfactory" | '
    '"good" | "excellent",\n'
    '  "domain_scores": {\n'
    '    "data_gathering":     {"raw": number, "max": number, "verdict": <verdict>},\n'
    '    "clinical_management":{"raw": number, "max": number, "verdict": <verdict>},\n'
    '    "interpersonal":      {"raw": number, "max": number, "verdict": <verdict>}\n'
    "  },\n"
    '  "criteria": [{\n'
    '    "criterion_id": string,\n'
    '    "domain": "data_gathering" | "clinical_management" | "interpersonal",\n'
    '    "verdict": "met" | "partially-met" | "missed",\n'
    '    "evidence": string,\n'
    '    "guideline_ref": string | null\n'
    "  }],\n"
    '  "safety_breach": null | {"what": string, "guideline_ref": string|null},\n'
    '  "highlights":   [string, ...] (1-3 entries),\n'
    '  "improvements": [string, ...] (1-3 entries),\n'
    '  "narrative":    string (1-2 paragraphs, spoken-aloud teaching debrief)\n'
    "}\n\n"
    "GRADING RULES:\n"
    "1. For every criterion in rubric.{data_gathering,clinical_management,"
    "interpersonal}, decide met/partially-met/missed using the criterion's "
    "`evidence` field as your match key. Quote the trainee directly in the "
    "criterion's `evidence` field of your output (your own observation, "
    "not the rubric's evidence string).\n"
    "2. Compute domain_scores: raw = sum(weight) for met (×1.0) + "
    "partially-met (×0.5); max = sum of all weights in that domain.\n"
    "3. Verdict bands by ratio raw/max: ≥0.85 excellent, ≥0.70 good, "
    "≥0.55 satisfactory, ≥0.40 borderline, otherwise clear-fail. Apply "
    "the same bands to the total across all three domains for "
    "global_rating.\n"
    "4. CITE, DO NOT INVENT. Every clinical_management criterion's "
    "guideline_ref MUST appear in registry_slice.guidelines. If the rubric "
    "criterion has no guideline_ref AND no rec applies, drop the criterion "
    "from your output rather than fabricating one.\n"
    "5. Set safety_breach only if the trainee did something dangerous "
    "(contraindicated drug, missed red flag, no safety-netting on a "
    "high-risk diagnosis). The narrative must lead with this regardless "
    "of the score.\n"
    "6. Pick 1–3 highlights and 1–3 improvements. Be specific.\n"
    "7. The narrative is a 1–2 paragraph teaching debrief in the voice of "
    "a senior clinician. No praise sandwiches, no sycophancy.\n"
)


def _build_attending_prompt(request: DebriefRequest) -> tuple[str, str]:
    user_payload = {
        "case_id": request.case_id,
        "case_summary": request.case_summary,
        "rubric": request.rubric,
        "registry_slice": request.registry_slice,
        "encounter_log": request.encounter_log,
    }
    return _ATTENDING_SYSTEM, json.dumps(user_payload, ensure_ascii=False)


def _parse_evaluation(text: str, request: DebriefRequest) -> CaseEvaluation:
    """Parse the model's JSON output into a CaseEvaluation, validating
    that all guideline_refs resolve in registry_slice."""
    cleaned = text.strip()
    # Strip code fences if the model ignored the instruction.
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"ollama attending output is not valid JSON: {e}") from e

    valid_refs = _collect_valid_refs(request.registry_slice)
    criteria = []
    for raw in data.get("criteria", []) or []:
        ref = raw.get("guideline_ref")
        if ref is not None and ref not in valid_refs:
            # Rule 4: drop fabricated refs rather than passing them through.
            _log.warning(
                "dropping criterion %s with unknown guideline_ref %r",
                raw.get("criterion_id"), ref,
            )
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


def _collect_valid_refs(registry_slice: dict[str, Any]) -> set[str]:
    """Collect every `<guideline_id>:<rec_id>` that appears in the
    registry slice. Used to drop fabricated refs from model output."""
    refs: set[str] = set()
    guidelines = registry_slice.get("guidelines") or []
    for g in guidelines:
        gid = g.get("id")
        if not gid:
            continue
        for rec in g.get("recommendations") or []:
            rid = rec.get("id")
            if rid:
                refs.add(f"{gid}:{rid}")
    return refs
