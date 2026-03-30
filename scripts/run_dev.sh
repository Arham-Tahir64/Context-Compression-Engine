#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$SCRIPT_DIR/.."

cd "$ROOT"

if [ -f ".env" ]; then
  export $(grep -v '^#' .env | xargs)
fi

uvicorn cce.main:app \
  --host "${HOST:-127.0.0.1}" \
  --port "${PORT:-8765}" \
  --reload \
  --log-level info
