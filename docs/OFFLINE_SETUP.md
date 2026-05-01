# Offline Setup Guide

medkit's `offline-first` branch runs the entire simulator on a single
machine with no cloud dependencies:

- **LLM:** Ollama (`medgemma1.5:4b`)
- **STT:** faster-whisper
- **TTS:** Piper
- **Voice transport:** self-hosted LiveKit OSS

Cloud providers (Gemini, Claude) are optional fallbacks.

---

## Prerequisites

- Docker Desktop or Docker Engine 20.10+ with the `compose` plugin.
- 16 GB RAM minimum (32 GB recommended). CPU works; an NVIDIA GPU
  accelerates inference but is not required.
- ~10 GB free disk space (Ollama model + Piper voices + Whisper model +
  container layers).
- macOS, Linux, or Windows + WSL2.

---

## One-time setup

```bash
git checkout offline-first
./scripts/setup-offline.sh
```

`setup-offline.sh` is idempotent. It:

1. Verifies Docker is installed.
2. Generates `livekit/livekit.yaml` with a fresh API secret if it
   doesn't exist.
3. Writes `backend/.env.local` with offline defaults if it doesn't exist.
4. Spins up the Ollama container and runs `ollama pull medgemma1.5:4b`
   if the model isn't cached yet.
5. Pre-pulls Piper voices into the named Docker volume.
6. Pre-pulls the faster-whisper `small.en` model into its named volume.

If you already have `medgemma1.5:4b` cached on the host's Ollama
installation, the docker volume gets its own copy — they are independent.

---

## Start the stack

```bash
docker compose -f docker-compose.offline.yml up
```

After all services are healthy, open <http://localhost:5173>.

What's running:

| Service | Container | Port | Purpose |
|---|---|---|---|
| `frontend` | medkit-frontend | 5173 | Vite dev server (React UI) |
| `backend` | medkit-backend | 8787 | FastAPI (provider routing, /voice/token, /agent/local/debrief) |
| `voice-worker` | medkit-voice-worker | — | LiveKit agent (offline variant) |
| `ollama` | medkit-ollama | 11434 | LLM inference |
| `livekit` | medkit-livekit | 7880, 7881 | WebRTC transport (LiveKit OSS) |
| `whisper` | medkit-whisper | 5001 | faster-whisper STT sidecar |
| `piper` | medkit-piper | 5002 | Piper TTS sidecar |

---

## Stop the stack

```bash
docker compose -f docker-compose.offline.yml down
```

Add `-v` to also drop the named volumes (Ollama models, Piper voices,
Whisper cache) — useful for clean re-tests.

---

## GPU acceleration (NVIDIA only)

Install the [NVIDIA Container
Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/),
then start with the GPU overlay:

```bash
docker compose -f docker-compose.offline.yml -f docker-compose.gpu.yml up
```

This pins Ollama to your GPU and switches `WHISPER_DEVICE` to `cuda`.

---

## Switching to a cloud provider

Edit `backend/.env.local`:

### Gemini (optional fallback)

```bash
LLM_PROVIDER=gemini
LLM_FALLBACK_CHAIN=gemini,ollama
GEMINI_API_KEY=your_key
```

### Claude (last-priority fallback)

```bash
LLM_PROVIDER=claude
LLM_FALLBACK_CHAIN=claude,ollama
ANTHROPIC_API_KEY=your_key
LIVEKIT_MODE=cloud
LIVEKIT_URL=wss://your-livekit-cloud-host
LIVEKIT_API_KEY=...
LIVEKIT_API_SECRET=...
DEEPGRAM_API_KEY=...
CARTESIA_API_KEY=...
```

Restart the backend container:

```bash
docker compose -f docker-compose.offline.yml restart backend voice-worker
```

When `LLM_PROVIDER=claude`, the frontend automatically routes the
attending debrief through Anthropic's Managed Agents endpoint (so you'll
also need to run `POST /agent/bootstrap` once).

---

## Choosing a different Ollama model

Pull it inside the container, then point the backend at it:

```bash
docker exec medkit-ollama ollama pull qwen2.5:14b-instruct
```

In `backend/.env.local`:

```bash
OLLAMA_MODEL_PATIENT=medgemma1.5:4b      # keep small for low latency voice
OLLAMA_MODEL_ATTENDING=qwen2.5:14b-instruct  # heavier model for grading
```

Restart:

```bash
docker compose -f docker-compose.offline.yml restart backend voice-worker
```

Recommended attending models when you have spare VRAM:

- `qwen2.5:14b-instruct` — strong instruction following, ~9 GB
- `mixtral:8x7b` — robust reasoning, ~26 GB
- `llama3.1:70b` — best quality, requires 40+ GB VRAM

---

## Choosing a different Whisper model

Edit `backend/.env.local`:

```bash
WHISPER_MODEL=medium.en     # better accuracy, ~1.5 GB
# or
WHISPER_MODEL=large-v3      # best quality, ~3 GB, GPU recommended
```

Then re-run the pre-pull script:

```bash
WHISPER_MODEL=medium.en ./scripts/pull-whisper-model.sh
docker compose -f docker-compose.offline.yml restart whisper
```

---

## Choosing a different Piper voice

Voices live in the `piper-voices` named volume. To add a new one:

```bash
# List currently installed voices.
docker exec medkit-piper ls /models | grep .onnx

# Pull a new voice (example: Lessac high quality).
docker exec medkit-piper bash -c \
  "curl -fL -o /models/en_US-lessac-high.onnx       https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/high/en_US-lessac-high.onnx \
&& curl -fL -o /models/en_US-lessac-high.onnx.json  https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/high/en_US-lessac-high.onnx.json"
```

Override the persona → voice mapping with `PIPER_VOICE_MAP` (JSON):

```bash
PIPER_VOICE_MAP='{"adult_male":"en_US-ryan-medium","adult_female":"en_US-lessac-high"}'
```

---

## Troubleshooting

### `docker compose up` fails immediately

Run `docker compose -f docker-compose.offline.yml ps` and look for the
unhealthy service. Common causes:

- Port already in use (5173, 8787, 11434, 7880). Stop the host process
  or change the port mapping.
- LiveKit secret not generated — re-run `./scripts/setup-offline.sh`.

### Voice doesn't connect

```bash
docker logs medkit-voice-worker
```

Check that LiveKit registered the worker. If you see
`livekit-plugins-openai` import errors, rebuild:

```bash
docker compose -f docker-compose.offline.yml build voice-worker
```

### Patient persona is slow

`medgemma1.5:4b` runs at ~30 tok/s on M-series Macs and ~10 tok/s on
8-core x86 CPUs. For lower latency:

- Pull a smaller model (`gemma2:2b`).
- Enable GPU mode (NVIDIA only).
- Reduce `OLLAMA_MODEL_PATIENT` to a quantized 2B variant.

### Attending debrief returns malformed JSON

Small Gemma models occasionally drift outside the JSON schema.
Workarounds:

- Use `qwen2.5:14b-instruct` for the attending model (better JSON-mode
  reliability).
- Enable the cloud fallback chain so the backend automatically retries
  the next provider on parse failure.

### `health` says provider is `claude` but I want offline

`backend/.env.local` overrides only matter if the backend container
loaded it. Restart:

```bash
docker compose -f docker-compose.offline.yml restart backend
```

Then visit <http://localhost:8787/health> and confirm `providers.llm`.

---

## Verifying the install

```bash
# Backend health.
curl http://localhost:8787/health

# Ollama is reachable from the backend.
docker exec medkit-backend curl -s http://ollama:11434/api/tags

# Whisper sidecar.
docker exec medkit-backend curl -s http://whisper:5001/health

# Piper sidecar.
docker exec medkit-backend curl -s http://piper:5002/health
```

Open <http://localhost:5173>, accept a polyclinic patient, and speak.
The first request streams audio through the entire offline pipeline.

---

## Going back to cloud (Claude on `main`)

The `main` branch is unchanged — it still uses the original cloud-only
configuration. Switch back with:

```bash
git checkout main
```

The two branches share the same data files, so cases stay consistent.
