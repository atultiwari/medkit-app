# medkit voice worker — offline LiveKit agent.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl ca-certificates ffmpeg \
 && rm -rf /var/lib/apt/lists/*

COPY backend/voice_agent_requirements.txt /app/backend/voice_agent_requirements.txt
RUN pip install -r /app/backend/voice_agent_requirements.txt

# OpenAI-compatible plugin so the worker can talk to Ollama via /v1.
RUN pip install "livekit-plugins-openai>=1.0" "httpx[http2]>=0.27"

COPY backend /app/backend

ENV PYTHONPATH=/app

CMD ["python", "-u", "backend/voice_agent_local.py", "start"]
