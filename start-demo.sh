#!/bin/zsh

set -u
set -o pipefail

PROJECT_ROOT="$HOME/Documents/EyeDropShop/Costumer_Agent"
BRAIN_DIR="$PROJECT_ROOT/eyedropshop-brain"
APP_DIR="$PROJECT_ROOT/shop-chat-agent"

KEY_FILE="/tmp/eyedropshop_brain_customer_key"
LOG_DIR="/tmp/eyedropshop-demo"

BRAIN_LOG="$LOG_DIR/brain.log"
OLLAMA_LOG="$LOG_DIR/ollama.log"
SHOPIFY_LOG="$LOG_DIR/shopify.log"


OLLAMA_PID=""
BRAIN_PID=""

mkdir -p "$LOG_DIR"

cleanup() {
  echo ""
  echo "Stopping demo services..."

  if [[ -n "${BRAIN_PID:-}" ]]; then
    kill "$BRAIN_PID" 2>/dev/null || true
  fi

  if [[ -n "${OLLAMA_PID:-}" ]]; then
    kill "$OLLAMA_PID" 2>/dev/null || true
  fi
}

trap cleanup EXIT INT TERM

wait_for_service() {
  local url="$1"
  local name="$2"

  for attempt in {1..60}; do
    if curl -fsS "$url" >/dev/null 2>&1; then
      echo "✓ $name is ready"
      return 0
    fi

    sleep 1
  done

  echo "✗ $name did not start"
  return 1
}

# Create the shared Brain key when necessary.
if [[ ! -s "$KEY_FILE" ]]; then
  umask 077
  openssl rand -hex 32 > "$KEY_FILE"
fi

BRAIN_KEY="$(<"$KEY_FILE")"

# Start Ollama if it is not already running.
if curl -fsS "http://127.0.0.1:11434/api/tags" >/dev/null 2>&1; then
  echo "✓ Ollama is already running"
else
  echo "Starting Ollama..."

  ollama serve > "$OLLAMA_LOG" 2>&1 &
  OLLAMA_PID=$!

  wait_for_service \
    "http://127.0.0.1:11434/api/tags" \
    "Ollama" || exit 1
fi

# Load the model before the demonstration.
echo "Loading qwen3.5:9b..."

curl -fsS "http://127.0.0.1:11434/api/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5:9b",
    "prompt": "Hello",
    "stream": false,
    "keep_alive": "30m"
  }' >/dev/null

echo "✓ Ollama model is loaded"

# Start the Brain API if it is not already running.
if curl -fsS "http://127.0.0.1:8000/health" >/dev/null 2>&1; then
  echo "✓ EyeDropShop Brain is already running"
else
  echo "Starting EyeDropShop Brain..."

  (
    cd "$BRAIN_DIR" || exit 1

    export BRAIN_CUSTOMER_API_KEY="$BRAIN_KEY"

    exec conda run \
      --no-capture-output \
      -n eyedropshop-brain \
      uvicorn brain_api:app \
      --host 127.0.0.1 \
      --port 8000
  ) > "$BRAIN_LOG" 2>&1 &

  BRAIN_PID=$!

  wait_for_service \
    "http://127.0.0.1:8000/health" \
    "EyeDropShop Brain" || {
      echo "Check: $BRAIN_LOG"
      exit 1
    }
fi

# Configure the Shopify application.
cd "$APP_DIR" || exit 1

export BRAIN_API_URL="http://127.0.0.1:8000"
export BRAIN_CUSTOMER_API_KEY="$BRAIN_KEY"
export OLLAMA_URL="http://127.0.0.1:11434"
export OLLAMA_MODEL="qwen3.5:9b"
export USE_EYEDROPSHOP_BRAIN="true"

# Keep the conversation secret stable between demo sessions.
SESSION_SECRET_FILE="$PROJECT_ROOT/.chat_session_secret"

if [[ ! -s "$SESSION_SECRET_FILE" ]]; then
  (
    umask 077
    openssl rand -hex 32 > "$SESSION_SECRET_FILE"
  ) || exit 1
fi

chmod 600 "$SESSION_SECRET_FILE"

export CHAT_SESSION_SECRET="$(<"$SESSION_SECRET_FILE")"
export CHAT_ALLOWED_ORIGINS="https://eye-agent-test.myshopify.com"
export BRAIN_REQUEST_TIMEOUT_MS="240000"

# Shopify App Proxy provides the stable storefront endpoint:
#   /apps/eye-agent/...
# Shopify CLI updates the proxy destination to the current dev tunnel, so no
# temporary trycloudflare.com URL needs to be copied into theme settings.
: > "$SHOPIFY_LOG"

echo "Starting Shopify..."
echo "Press Control+C when the demonstration is finished."
echo ""

conda run \
  --no-capture-output \
  -n shop-chat-agent \
  shopify app dev --config shop-chat-agent --store eye-agent-test.myshopify.com \
    --theme-app-extension-port "${SHOPIFY_THEME_PORT:-9293}" 2>&1 |
  tee "$SHOPIFY_LOG"
