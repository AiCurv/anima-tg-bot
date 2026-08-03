# telegram-anima-diffusion-bot

Pure-Python Anima (NVIDIA Cosmos-Predict2-2B-Text2Image) Telegram bot.

**No ComfyUI. No Docker. No GPU.** Just Python + diffusers on a free GitHub Actions CPU runner.

## Architecture

```
  Telegram User
        │  /generate <prompt>
        ▼
  Cloudflare Worker  ── (instant 200, inline buttons)
        │  repository_dispatch
        ▼
  GitHub Actions (ubuntu-latest, 7GB RAM, CPU-only)
        │  run_anima.py
        │   ├── Stage 1: Qwen3 0.6B text encoder → embeddings → free
        │   ├── Stage 2: Anima transformer       → latents      → free
        │   └── Stage 3: Qwen-Image VAE          → output.png
        │
        ▼  curl sendPhoto
  Telegram User  ← image
```

## Files

```
.
├── run_anima.py                          # Pure Python staged inference (CPU)
├── requirements.txt                      # Pinned CPU-only deps
├── .github/workflows/anima_pipeline.yml  # GitHub Actions runner
├── worker/
│   ├── src/index.js                      # Cloudflare Worker (webhook router)
│   ├── wrangler.toml                     # Worker config
│   └── package.json
└── deploy.sh                             # One-shot deploy script
```

## Setup

### 1. Push this repo to GitHub

Already done — this repo lives at https://github.com/AiCurv/anima-tg-bot.

### 2. Add GitHub Actions secrets

In `Settings → Secrets and variables → Actions → New repository secret` (or via API):

| Secret name            | Value placeholder              |
|------------------------|--------------------------------|
| `TELEGRAM_BOT_TOKEN`   | your Telegram bot token        |
| `TELEGRAM_CHAT_ID`     | your Telegram user id          |
| `HF_TOKEN`             | your HuggingFace token         |

### 3. Deploy the Cloudflare Worker

```bash
cd worker

# Set env vars (one-time per shell) — replace with your actual values
export CLOUDFLARE_API_TOKEN="..."
export CLOUDFLARE_ACCOUNT_ID="..."
export TELEGRAM_BOT_TOKEN="..."
export ALLOWED_TELEGRAM_USER_ID="..."
export GITHUB_TOKEN="..."
export GITHUB_REPO="AiCurv/anima-tg-bot"

# Deploy + push secrets
npx wrangler deploy
echo "$TELEGRAM_BOT_TOKEN"      | npx wrangler secret put TELEGRAM_BOT_TOKEN
echo "$ALLOWED_TELEGRAM_USER_ID" | npx wrangler secret put ALLOWED_TELEGRAM_USER_ID
echo "$GITHUB_TOKEN"            | npx wrangler secret put GITHUB_TOKEN
echo "$GITHUB_REPO"             | npx wrangler secret put GITHUB_REPO
```

Worker URL will be `https://telegram-anima-bot.<your-subdomain>.workers.dev`.

### 4. Register Telegram webhook

```bash
curl "https://api.telegram.org/bot<TOKEN>/setWebhook?url=https://telegram-anima-bot.<your-subdomain>.workers.dev/webhook"
```

### 5. Test it

Open Telegram, message the bot:

```
/generate 1girl, anime, vibrant, masterpiece
```

The bot will instantly reply with an inline-keyboard message ("🔍 Check Status" / "❌ Cancel Run"), then ~10-15 minutes later the generated image will arrive.

## How memory staging works (the 7GB trick)

GitHub Actions ubuntu-latest gives ~7GB RAM. The full Anima model is ~4GB on disk (8GB in fp32 RAM). We cannot load all three sub-models at once, so `run_anima.py` stages them:

| Stage | Model loaded | RAM peak | Output | Then |
|-------|--------------|----------|--------|------|
| 1 | Qwen3 0.6B text encoder  | ~1.5 GB | prompt embeddings (`/tmp/anima_prompt_embeds.pt`) | `del` + `gc.collect()` |
| 2 | Anima transformer (2B)   | ~5 GB  | latents (`/tmp/anima_latents.pt`)                  | `del` + `gc.collect()` |
| 3 | Qwen-Image VAE           | ~1 GB  | `output.png`                                       | done |

## Why no ComfyUI / Docker?

- ComfyUI pulls in a whole GUI server, workflows API, custom-node ecosystem — adds 3+ GB to the runner and never fit in 7GB RAM with the model loaded.
- Docker adds another layer of indirection and cache misses for the HF hub cache.
- Pure Python + `diffusers.from_pretrained` is the smallest possible footprint and gives us perfect control over `gc.collect()` between stages.

## Tuning

Edit the defaults in `.github/workflows/anima_pipeline.yml`:

| Param     | Default | Notes |
|-----------|---------|-------|
| `--width` / `--height` | 1024 | 512² for faster test, max 1536² |
| `--steps`  | 30     | 4-5 for turbo, 30-50 for base |
| `--cfg`    | 4.5    | 1.0 for turbo, 4-5 for base |
| `--seed`   | -1     | -1 = random |

For the **turbo** model (`anima-turbo-v1.0.safetensors` instead of `anima-base-v1.0.safetensors`), edit `run_anima.py`:

```python
DIFFUSION_WEIGHT_FILE = "split_files/diffusion_models/anima-turbo-v1.0.safetensors"
```

and in the workflow use `--steps 8 --cfg 1.0`.

## Debugging

- **Workflow log**: Actions tab → click the run → expand "Run Anima generation"
- **Worker log**: `cd worker && npx wrangler tail`
- **Check Telegram webhook**: `curl "https://api.telegram.org/bot<TOKEN>/getWebhookInfo"`
