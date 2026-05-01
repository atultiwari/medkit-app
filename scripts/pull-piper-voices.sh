#!/usr/bin/env bash
# Pull Piper voices into the named volume `piper-voices`.
#
# Idempotent: skips voices already present in the volume. Voices come from
# Hugging Face under rhasspy/piper-voices.
#
# Override the list by passing a destination directory as $1; otherwise we
# write into a one-shot container that mounts the named volume.

set -euo pipefail

VOICES=(
  "en_US-ryan-medium"
  "en_US-amy-medium"
  "en_US-joe-medium"
  "en_US-lessac-medium"
  "en_US-kathleen-low"
)

BASE_URL="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US"

# Map each voice to its locale subdir (en/en_US/<speaker>/<quality>/).
declare -A SUBDIRS=(
  [en_US-ryan-medium]="ryan/medium"
  [en_US-amy-medium]="amy/medium"
  [en_US-joe-medium]="joe/medium"
  [en_US-lessac-medium]="lessac/medium"
  [en_US-kathleen-low]="kathleen/low"
)

# When called inside the piper container, $1 is the on-disk voice dir.
# When called from the host, we run a one-shot container that mounts the
# named volume.
if [[ $# -gt 0 && -d "$1" ]]; then
  DEST="$1"
  echo "Writing voices into $DEST"
  for name in "${VOICES[@]}"; do
    if [[ -f "$DEST/$name.onnx" && -f "$DEST/$name.onnx.json" ]]; then
      echo "  $name already present — skipping"
      continue
    fi
    sub="${SUBDIRS[$name]}"
    echo "  fetching $name"
    curl -fL --progress-bar -o "$DEST/$name.onnx" \
      "$BASE_URL/$sub/$name.onnx"
    curl -fL --progress-bar -o "$DEST/$name.onnx.json" \
      "$BASE_URL/$sub/$name.onnx.json"
  done
  echo "Done."
  exit 0
fi

# Host-side path: spin a one-shot container mounting the named volume.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/.." &>/dev/null && pwd)"

docker run --rm \
  -v "$ROOT_DIR/scripts:/scripts:ro" \
  -v "medkit-offline_piper-voices:/models" \
  curlimages/curl:8.10.1 \
  /scripts/pull-piper-voices.sh /models
