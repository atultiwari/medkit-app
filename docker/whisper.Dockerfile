# Whisper sidecar — minimal FastAPI server wrapping faster-whisper.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/hf

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

RUN pip install \
      "faster-whisper>=1.0" \
      "fastapi>=0.110" \
      "uvicorn>=0.30"

COPY docker/whisper_server.py /app/whisper_server.py

EXPOSE 5001

CMD ["uvicorn", "whisper_server:app", "--host", "0.0.0.0", "--port", "5001"]
