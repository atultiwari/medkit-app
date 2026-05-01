"""Local attending debrief — drop-in replacement for Anthropic Managed Agents.

When `LLM_PROVIDER != claude`, the frontend's event-stream URL is rewritten
to `/agent/local/debrief`. This module produces an SSE event stream that
mirrors the shape `eventStreamRenderer.tsx` already consumes:

  event: agent.started     data: {"session_id": "..."}
  event: agent.text_delta  data: {"text": "..."}                   (optional)
  event: agent.custom_tool_use  data: {"name":"render_case_evaluation","input":{...}}
  event: agent.completed   data: {"reason":"end_turn"}
  event: proxy_error       data: {"message":"..."}                  (on failure)

The grading itself is a single call into `LLMProvider.grade_attending`. We
emit the resulting `CaseEvaluation` as one synthetic `agent.custom_tool_use`
event so the existing `<CaseEvaluationCard>` renderer just works.
"""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
from dataclasses import asdict
from typing import Any, AsyncIterator

from backend.providers import (
    DebriefRequest,
    LLMProvider,
    get_llm_provider,
)

_log = logging.getLogger("medkit.agent_local")


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


async def run_local_debrief(
    request: DebriefRequest,
    provider: LLMProvider | None = None,
) -> AsyncIterator[str]:
    """Yield SSE-formatted strings for one local debrief run."""
    session_id = f"local-{secrets.token_hex(6)}"
    provider = provider or get_llm_provider()
    yield _sse(
        "agent.started",
        {"session_id": session_id, "provider": provider.name},
    )
    try:
        evaluation = await provider.grade_attending(request)
    except asyncio.CancelledError:
        raise
    except Exception as e:
        _log.exception("local debrief failed")
        yield _sse("proxy_error", {"message": str(e)})
        return

    payload = asdict(evaluation)
    # The frontend's eventStreamRenderer expects the canonical shape:
    #   { type: 'agent.custom_tool_use', name: 'render_case_evaluation',
    #     input: { ... } }
    # Emit the input shape exactly as `customTools.ts` Zod-validates.
    tool_use = {
        "type": "agent.custom_tool_use",
        "id": f"tool-{secrets.token_hex(6)}",
        "name": "render_case_evaluation",
        "input": _to_renderer_shape(payload),
    }
    yield _sse("agent.custom_tool_use", tool_use)
    yield _sse("agent.completed", {"reason": "end_turn"})


def _to_renderer_shape(eval_dict: dict[str, Any]) -> dict[str, Any]:
    """Match the Zod schema `RenderCaseEvaluationInput` in
    `src/agents/customTools.ts`. The dataclass already uses the right
    keys; we just drop None safety_breach so the optional field is
    omitted instead of explicitly null."""
    out = dict(eval_dict)
    if out.get("safety_breach") is None:
        out.pop("safety_breach", None)
    return out
