# medkit FastAPI backend.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install -r /app/backend/requirements.txt

# Provider deps that the offline path needs (httpx is already pulled by
# fastapi/anthropic; faster-whisper only required for in-process STT).
RUN pip install "httpx[http2]>=0.27"

COPY backend /app/backend

ENV PYTHONPATH=/app
EXPOSE 8787

# uvicorn directly so we get clean signal handling.
CMD ["uvicorn", "backend.server:app", "--host", "0.0.0.0", "--port", "8787", "--log-level", "info"]
