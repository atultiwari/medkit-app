# Piper TTS sidecar — minimal FastAPI server wrapping the piper-tts package.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl ca-certificates wget \
 && rm -rf /var/lib/apt/lists/*

# piper-tts is the python package with the synthesise() entry point;
# onnxruntime is its inference backend.
RUN pip install \
      "piper-tts>=1.2" \
      "onnxruntime>=1.18" \
      "fastapi>=0.110" \
      "uvicorn>=0.30"

COPY docker/piper_server.py /app/piper_server.py
COPY scripts/pull-piper-voices.sh /app/pull-piper-voices.sh

# Voices land in /models on first boot; if they're already mounted via the
# named volume we skip the download.
ENV PIPER_VOICE_DIR=/models

EXPOSE 5002

CMD ["bash", "-lc", "/app/pull-piper-voices.sh /models && uvicorn piper_server:app --host 0.0.0.0 --port 5002"]
