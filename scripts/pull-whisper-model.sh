#!/usr/bin/env bash
# Pre-pull faster-whisper model into the named volume so the first
# /transcribe request doesn't block on a large download.
#
# Default: small.en. Override via WHISPER_MODEL env.

set -euo pipefail

WHISPER_MODEL="${WHISPER_MODEL:-small.en}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/.." &>/dev/null && pwd)"

echo "Pulling faster-whisper model: $WHISPER_MODEL"

# Run the whisper image with a one-shot Python that just constructs the
# WhisperModel — this triggers the snapshot download into the cache.
docker run --rm \
  -e WHISPER_MODEL="$WHISPER_MODEL" \
  -v "medkit-offline_whisper-models:/models" \
  -e HF_HOME=/models/hf \
  python:3.12-slim \
  bash -lc "pip install --quiet 'faster-whisper>=1.0' && python -c 'from faster_whisper import WhisperModel; WhisperModel(\"$WHISPER_MODEL\", device=\"cpu\", compute_type=\"int8\")'"

echo "Done."
