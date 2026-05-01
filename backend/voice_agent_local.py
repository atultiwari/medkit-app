"""medkit voice worker — fully offline.

Drop-in replacement for `voice_agent.py`. Joins every LiveKit room created
by the FastAPI server and roleplays the patient using:

    Browser mic → faster-whisper STT → Ollama (medgemma1.5:4b) → Piper TTS → Browser

The persona prompt and voice gender come from room metadata (set by the
backend `/voice/token` endpoint when the room is created).

Run:
    backend/.venv-voice/bin/python backend/voice_agent_local.py dev

LiveKit's `livekit.plugins.openai` plugin works against any OpenAI-compatible
chat endpoint, including Ollama (`http://ollama:11434/v1`). For STT and TTS
we use the project's own `WhisperProvider` and `PiperProvider` via the
`livekit.plugins.silero` VAD + custom STT/TTS adapters.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

# Allow `backend.providers` / `backend.livekit_plugins` imports when launched
# directly. PYTHONPATH already includes /app under the Docker entrypoint.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv
from livekit import agents, rtc
from livekit.agents import (
    Agent,
    AgentSession,
    RoomInputOptions,
    WorkerOptions,
    cli,
)
from livekit.plugins import silero

# OpenAI-compatible plugin used to talk to Ollama via its /v1 endpoint.
try:  # pragma: no cover — optional plugin depending on env
    from livekit.plugins import openai as lk_openai  # type: ignore
except Exception:  # pragma: no cover
    lk_openai = None  # surfaced clearly at runtime if missing

# Local STT/TTS plugins (custom — see backend/livekit_plugins/).
from backend.livekit_plugins.whisper_stt import WhisperSTT
from backend.livekit_plugins.piper_tts import PiperTTS

# Load .env.local first (project convention), .env as fallback.
_BACKEND = Path(__file__).resolve().parent
load_dotenv(_BACKEND / ".env.local")
load_dotenv(_BACKEND / ".env")

logger = logging.getLogger("medkit.voice-agent-local")
logger.setLevel(logging.INFO)


# Voice mapping for the Piper sidecar. Same gender → adult voice;
# pediatric cases use the parent voice (the persona prompt routes the
# agent's first-person speaker through the parent already).
VOICE_BY_GENDER = {
    "M": "adult_male",
    "F": "adult_female",
}
PARENT_BY_GENDER = {
    "M": "parent_male",
    "F": "parent_female",
}

DEFAULT_INSTRUCTIONS = (
    "You are a patient speaking to a doctor. Keep replies to 1-2 short "
    "spoken sentences. Output spoken dialogue only — no stage directions, "
    "no asterisks."
)
DEFAULT_INITIAL = "Hi doc."


def _hash_str(s: str) -> int:
    """FNV-1a — mirrors `src/voice/patientPersona.ts` so TS and Python pick
    the same voice slot for the same case ID."""
    h = 0x811C9DC5
    for ch in s:
        h ^= ord(ch)
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


def parse_metadata(raw: str | None) -> dict:
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("room metadata is not valid JSON: %r", raw[:120])
        return {}


def _build_llm():
    """Construct the LiveKit-side LLM. Defaults to Ollama via the OpenAI-
    compatible plugin; falls back to error if the plugin is not installed."""
    if lk_openai is None:
        raise RuntimeError(
            "livekit-plugins-openai is required for offline voice. "
            "Install with: pip install 'livekit-plugins-openai'"
        )
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    model = os.environ.get("OLLAMA_MODEL_PATIENT", "medgemma1.5:4b")
    # The OpenAI plugin needs an api_key — Ollama ignores it but the
    # SDK requires a non-empty value.
    return lk_openai.LLM(
        model=model,
        base_url=f"{base_url}/v1",
        api_key="ollama",
        temperature=0.8,
    )


async def entrypoint(ctx: agents.JobContext):
    await ctx.connect()

    meta = parse_metadata(ctx.room.metadata)
    case_id = meta.get("caseId") or meta.get("case_id") or "unknown"
    speaker_gender = (meta.get("voiceGender") or meta.get("gender") or "M").upper()
    is_pediatric = bool(meta.get("isPediatric") or meta.get("pediatric"))
    system_prompt = meta.get("systemPrompt") or DEFAULT_INSTRUCTIONS
    initial_line = meta.get("initialLine") or DEFAULT_INITIAL

    voice_alias = (
        PARENT_BY_GENDER.get(speaker_gender, "parent_female")
        if is_pediatric
        else VOICE_BY_GENDER.get(speaker_gender, "adult_male")
    )

    logger.info(
        "joining room=%s case=%s gender=%s pediatric=%s voice=%s",
        ctx.room.name, case_id, speaker_gender, is_pediatric, voice_alias,
    )

    session = AgentSession(
        stt=WhisperSTT(),
        llm=_build_llm(),
        tts=PiperTTS(voice=voice_alias),
        vad=silero.VAD.load(),
    )

    agent = Agent(instructions=system_prompt)

    await session.start(
        agent=agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(),
    )

    FAREWELLS = [
        "Thank you, doctor. Take care.",
        "Okay, thanks doc. Goodbye.",
        "Thanks for your help. Bye.",
        "Alright, take care. Goodbye.",
    ]
    farewell_pick = FAREWELLS[_hash_str(case_id) % len(FAREWELLS)]

    @ctx.room.local_participant.register_rpc_method("farewell")
    async def _on_farewell(data: rtc.RpcInvocationData) -> str:
        logger.info("rpc farewell invoked by %s", data.caller_identity)
        try:
            session.say(farewell_pick)
        except Exception as e:
            logger.exception("session.say failed: %s", e)
            return "error"
        return "ok"

    await session.generate_reply(
        instructions=(
            f'Stay strictly in character. Speak this opening line, naturally, '
            f'as the patient (or accompanying parent for pediatric cases) '
            f'arriving in the room: "{initial_line}". One short sentence only.'
        ),
    )


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(entrypoint_fnc=entrypoint, agent_name="medkit-voice")
    )
