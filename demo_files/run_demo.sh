#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Run from the repo root:  bash demo_files/run_demo.sh
# ─────────────────────────────────────────────────────────────────────────────

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"

echo ""
echo "══════════════════════════════════════════════════════"
echo "  Insurance MAS Demo — Starting Up"
echo "══════════════════════════════════════════════════════"
echo ""

# ── Check .env ─────────────────────────────────────────────────────────────
if [ ! -f "$ROOT/.env" ]; then
  echo "  ERROR: No .env file found at $ROOT/.env"
  echo "  Please create it with MONGODB_URL and PINECONE_API_KEY"
  exit 1
fi
echo "  .env file found"

# ── Install demo dependencies ───────────────────────────────────────────────
echo "  Installing/checking Flask dependencies…"
pip install flask flask-cors --quiet

echo "  Dependencies ready"

# ── Check Ollama ────────────────────────────────────────────────────────────
if command -v ollama &> /dev/null; then
  echo "  Ollama detected"
else
  echo "  Ollama not found in PATH — make sure it is running"
fi

echo ""
echo "  Starting server on http://localhost:5050"
echo "  Open this URL in your browser."
echo "  Press Ctrl+C to stop."
echo ""

cd "$ROOT"
python demo_files/app.py