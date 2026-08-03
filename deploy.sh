#!/usr/bin/env bash
# =============================================================================
# telegram-anima-diffusion-bot :: deploy.sh
# =============================================================================
# One-shot deploy script. Run from repo root.
#
#   ./deploy.sh
#
# Requires: wrangler (npm i -g wrangler) and the Cloudflare token set.
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
WORKER_DIR="$REPO_ROOT/worker"

echo "============================================================"
echo "  Deploying Cloudflare Worker (telegram-anima-bot)"
echo "============================================================"

# --- 1. Verify env vars (tokens come from deploy-time env) ---
: "${CLOUDFLARE_API_TOKEN:?Need CLOUDFLARE_API_TOKEN}"
: "${CLOUDFLARE_ACCOUNT_ID:?Need CLOUDFLARE_ACCOUNT_ID}"
: "${TELEGRAM_BOT_TOKEN:?Need TELEGRAM_BOT_TOKEN}"
: "${ALLOWED_TELEGRAM_USER_ID:?Need ALLOWED_TELEGRAM_USER_ID}"
: "${GITHUB_TOKEN:?Need GITHUB_TOKEN}"
: "${GITHUB_REPO:?Need GITHUB_REPO e.g. user/anima-tg-bot}"

cd "$WORKER_DIR"

export CLOUDFLARE_API_TOKEN
export CLOUDFLARE_ACCOUNT_ID

# --- 2. Install wrangler if missing ---
if ! command -v wrangler >/dev/null 2>&1; then
  echo "Installing wrangler..."
  npm i -g wrangler
fi

# --- 3. Deploy worker ---
echo "Deploying worker..."
wrangler deploy

# --- 4. Push secrets ---
echo "Pushing secrets..."
echo "$TELEGRAM_BOT_TOKEN"      | wrangler secret put TELEGRAM_BOT_TOKEN
echo "$ALLOWED_TELEGRAM_USER_ID" | wrangler secret put ALLOWED_TELEGRAM_USER_ID
echo "$GITHUB_TOKEN"            | wrangler secret put GITHUB_TOKEN
echo "$GITHUB_REPO"             | wrangler secret put GITHUB_REPO

# --- 5. Show worker URL ---
echo ""
echo "============================================================"
echo "  Worker deployed. Find your URL in wrangler output above."
echo ""
echo "  Next: register Telegram webhook:"
echo "    curl \"https://api.telegram.org/bot\${TELEGRAM_BOT_TOKEN}/setWebhook?url=<WORKER_URL>/webhook\""
echo "============================================================"
