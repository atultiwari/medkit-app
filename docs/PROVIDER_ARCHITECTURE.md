# Provider Architecture

The `offline-first` branch introduces a provider abstraction so the
simulator can swap LLM, STT, and TTS implementations at runtime without
changing UI or game code.

---

## High-level flow

```
┌─────────────────┐    /agent/patient/stream   ┌──────────────────┐    Ollama / Gemini / Claude
│  Browser (UI)   │ ─────────────────────────► │  FastAPI backend │ ───────────────────────────►
│                 │                            │                  │
│  /voice/token   │ ◄───────────────────────── │  Provider chain  │ ◄───────────────────────────
└─────────────────┘                            └──────────────────┘    Whisper / Piper sidecars
        │                                              │
        │   WebRTC (LiveKit local OSS)                 │
        ▼                                              ▼
┌─────────────────┐                            ┌──────────────────┐
│  LiveKit room   │ ◄───────────────────────── │  Voice worker    │
│                 │                            │  (offline agent) │
└─────────────────┘                            └──────────────────┘
```

---

## Backend

### `backend/providers/`

Three abstract base classes (`base.py`) plus their implementations:

| Capability | Default | Optional |
|---|---|---|
| `LLMProvider` | `OllamaProvider` (`medgemma1.5:4b`) | `GeminiProvider`, `ClaudeProvider` |
| `STTProvider` | `WhisperProvider` (faster-whisper) | `DeepgramProvider` |
| `TTSProvider` | `PiperProvider` | `CartesiaProvider` |

The factory (`factory.py`) reads env vars and returns a singleton per
capability. `get_chain()` returns an ordered fallback list so the
backend can demote to the next provider when the primary fails.

### Capability contract

```python
class LLMProvider(ABC):
    @property
    def supports_tools(self) -> bool: ...

    async def stream_chat(messages, system, max_tokens, temperature
        ) -> AsyncIterator[ChatDelta]: ...

    async def grade_attending(request: DebriefRequest
        ) -> CaseEvaluation: ...
```

`grade_attending` MUST drop any `guideline_ref` that does not appear in
`request.registry_slice`. Hallucinated cites are silently rewritten to
`null` rather than passed to the UI.

### Endpoints

| Route | Provider used | Notes |
|---|---|---|
| `POST /agent/patient/stream` | LLM (full fallback chain) | SSE `{text}` deltas, `{done:true}` terminator |
| `POST /agent/local/debrief` | LLM (primary only) | SSE events mirroring Managed Agents shape |
| `POST /agent/sessions/.../events` and friends | Anthropic SDK directly | Only used when `LLM_PROVIDER=claude` |
| `POST /voice/token` | LiveKit (local or cloud) | Mints JWT + creates room with metadata |

### Voice worker

`backend/voice_agent.py` (cloud) and `backend/voice_agent_local.py`
(offline) are mutually exclusive — the compose stack only runs the local
variant. Both join LiveKit rooms created by `/voice/token` and roleplay
the patient. The local worker uses:

- `WhisperSTT` (`backend/livekit_plugins/whisper_stt.py`) — wraps the
  HTTP sidecar.
- `livekit-plugins-openai` LLM, pointed at `OLLAMA_BASE_URL/v1`.
- `PiperTTS` (`backend/livekit_plugins/piper_tts.py`) — wraps the HTTP
  sidecar.
- `silero.VAD` — same as the cloud worker.

### Provider env vars (full list)

See `backend/.env.example`.

---

## Frontend

The browser doesn't know which provider answers — everything is mediated
through the backend. Two modules read provider config:

### `src/voice/providers/index.ts`

Calls `/health` once at startup, caches the result, and exposes:

- `getProviderConfig()` — async, populates the cache.
- `getProviderConfigSync()` — synchronous read after the cache is warm.
- `isManagedAgentMode(config?)` — true iff `LLM_PROVIDER=claude`.

### `src/agents/useAttendingDebrief.ts`

Branches on `isManagedAgentMode`:

- **Cloud (Claude):** bootstrap → createSession → sendUserMessage →
  openEventStream. Same path as `main`.
- **Offline (Ollama / Gemini):** post `DebriefRequest` to
  `/agent/local/debrief` and consume the SSE stream via
  `openLocalDebriefStream`.

Both paths produce the same `CaseEvaluation` payload, so the existing
`<CaseEvaluationCard>` renderer is unchanged.

---

## Adding a new provider

1. **Backend:** create `backend/providers/<name>_provider.py`,
   implement `LLMProvider` (or STT/TTS), import-wire it in
   `factory.py`'s `_build_llm` (or stt/tts).
2. **Tests:** add a smoke test under `backend/tests/test_<name>_provider.py`.
3. **Env:** document the new env vars in `backend/.env.example`.
4. **Frontend type:** extend the union in
   `src/voice/providers/types.ts` if the name is user-visible.
5. **Docs:** update `docs/OFFLINE_SETUP.md` and this file.

The UI does not need any changes; the backend's `/health` already lists
the active provider name and the frontend just reads it.

---

## Failover semantics

`/agent/patient/stream` walks the chain on hard errors (HTTP 5xx,
network timeout, malformed stream). The browser sees a unified SSE
stream — only the *final* fallback's error reaches the user.

`/agent/local/debrief` does NOT auto-failover today; it uses
`get_llm_provider()` (primary only) because attending grading is a
single high-stakes call. Add chain-walking there only after measuring
real failure rates per provider.
