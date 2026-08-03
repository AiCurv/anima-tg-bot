#!/usr/bin/env python3
"""
================================================================================
 telegram-anima-diffusion-bot  ::  run_anima.py
================================================================================
 Pure-Python Anima (NVIDIA Cosmos-Predict2-2B-Text2Image) inference runner.

 Architecture (custom — Anima uses non-stock components):
   - Text Encoder: Qwen3 0.6B base (provides cross-attn context for llm_adapter)
   - Transformer:  Anima (custom 28-block DiT with integrated llm_adapter)
       Loaded via custom PyTorch module (anima_model.py) that matches the
       original ComfyUI-format weight names exactly.
   - VAE:          Qwen-Image VAE (AutoencoderKLQwenImage)

 Memory staging (CRITICAL for 7GB RAM CPU runner):
   Stage 1 : Load Qwen3 + tokenize  -> save hidden states  -> free
   Stage 2 : Load Anima Transformer -> denoise              -> free
   Stage 3 : Load VAE               -> decode               -> save image

 NO ComfyUI. NO Docker. NO GPU. Pure Python + diffusers + custom model.
================================================================================
"""

import os
import sys
import gc
import time
import shutil
import argparse
import traceback
from pathlib import Path

import torch

# Make sure anima_model.py is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from anima_model import load_anima_transformer, AnimaTransformer, weight_dtype_next

# ---------------------------------------------------------------------------
#  Banner / log
# ---------------------------------------------------------------------------
def banner(msg):
    bar = "=" * 70
    print(f"\n{bar}\n  {msg}\n{bar}", flush=True)

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# ---------------------------------------------------------------------------
#  Telegram helpers
# ---------------------------------------------------------------------------
def tg_send_message(bot_token, chat_id, text):
    if not bot_token or not chat_id:
        return
    try:
        import requests
        requests.post(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            json={"chat_id": int(chat_id), "text": text, "parse_mode": "Markdown"},
            timeout=15,
        )
    except Exception as e:
        log(f"  ! tg message failed: {e}")

def tg_send_photo(bot_token, chat_id, image_path, caption=""):
    if not bot_token or not chat_id:
        return False
    try:
        import requests
        with open(image_path, "rb") as f:
            r = requests.post(
                f"https://api.telegram.org/bot{bot_token}/sendPhoto",
                data={"chat_id": int(chat_id), "caption": caption[:1000]},
                files={"photo": f},
                timeout=120,
            )
        return r.status_code == 200
    except Exception as e:
        log(f"  ! tg photo failed: {e}")
        return False

# ---------------------------------------------------------------------------
#  Hugging Face helpers
# ---------------------------------------------------------------------------
HF_TOKEN = os.environ.get("HF_TOKEN", None)

def hf_get(repo_id, filename):
    from huggingface_hub import hf_hub_download
    log(f"  -> hf: {repo_id} / {filename}")
    return hf_hub_download(repo_id=repo_id, filename=filename, token=HF_TOKEN)

def build_local_component(name, spec):
    """Build a local directory with files needed for from_pretrained()."""
    local_dir = Path(f"/tmp/anima_components/{name}")
    local_dir.mkdir(parents=True, exist_ok=True)
    for item in spec:
        if "local_path" in item:
            src = Path(item["local_path"])
            dst_name = item.get("rename_as") or src.name
            dst = local_dir / dst_name
            if not dst.exists():
                shutil.copy(src, dst)
                log(f"  -> staged (local): {dst_name}")
        else:
            src = hf_get(item["repo"], item["file"])
            dst_name = item.get("rename_as") or Path(item["file"]).name
            dst = local_dir / dst_name
            if not dst.exists():
                shutil.copy(src, dst)
                log(f"  -> staged: {dst_name}")
    return str(local_dir)

# ---------------------------------------------------------------------------
#  Pipeline configuration
# ---------------------------------------------------------------------------
ANIMA_REPO        = "circlestone-labs/Anima"
QWEN3_BASE_REPO   = "Qwen/Qwen3-0.6B-Base"
QWEN_IMAGE_REPO   = "Qwen/Qwen-Image"

# Use turbo for CPU (8 steps, CFG 1)
DIFFUSION_WEIGHT_FILE = "split_files/diffusion_models/anima-turbo-v1.0.safetensors"
ENCODER_WEIGHT_FILE   = "split_files/text_encoders/qwen_3_06b_base.safetensors"
VAE_WEIGHT_FILE       = "split_files/vae/qwen_image_vae.safetensors"

# ---------------------------------------------------------------------------
#  Stage 1: Qwen3 text encoding
# ---------------------------------------------------------------------------
def stage1_encode_prompt(prompt, negative_prompt, device, dtype):
    """Stage 1 - Run Qwen3 over prompt to get hidden states + token IDs."""
    banner("STAGE 1 / 3  ::  Text Encoder (Qwen3 0.6B)")

    from transformers import AutoTokenizer, AutoModelForCausalLM

    local_dir = build_local_component("text_encoder", [
        {"repo": ANIMA_REPO,      "file": ENCODER_WEIGHT_FILE, "rename_as": "model.safetensors"},
        {"local_path": "configs/text_encoder/config.json"},
        {"repo": QWEN3_BASE_REPO, "file": "tokenizer_config.json"},
        {"repo": QWEN3_BASE_REPO, "file": "tokenizer.json"},
        {"repo": QWEN3_BASE_REPO, "file": "vocab.json"},
        {"repo": QWEN3_BASE_REPO, "file": "merges.txt"},
        {"repo": QWEN3_BASE_REPO, "file": "generation_config.json"},
    ])

    log("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(local_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log("Loading Qwen3 0.6B text encoder...")
    text_encoder = AutoModelForCausalLM.from_pretrained(
        local_dir,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    text_encoder.eval()
    log(f"  qwen3 params: {sum(p.numel() for p in text_encoder.parameters())/1e6:.1f}M")

    def encode(text):
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        ).to(device)
        with torch.no_grad():
            out = text_encoder(**inputs, output_hidden_states=True)
            # Use last hidden state as text features
            hidden = out.hidden_states[-1]
        return hidden, inputs.input_ids

    log(f"Encoding prompt: {prompt[:80]}")
    prompt_hidden, prompt_ids = encode(prompt)

    if negative_prompt:
        neg_hidden, neg_ids = encode(negative_prompt)
    else:
        # For CFG=1 we don't really need negatives, but keep shape consistent
        neg_hidden = torch.zeros_like(prompt_hidden)
        neg_ids = prompt_ids

    log(f"  prompt_hidden shape: {prompt_hidden.shape}")
    log(f"  prompt_ids shape:    {prompt_ids.shape}")

    # Save to disk so we can free Qwen3
    embeds_path = "/tmp/anima_prompt_embeds.pt"
    torch.save({
        "prompt_hidden":   prompt_hidden.cpu(),
        "prompt_ids":      prompt_ids.cpu(),
        "negative_hidden": neg_hidden.cpu(),
        "negative_ids":    neg_ids.cpu(),
    }, embeds_path)

    log("Freeing Qwen3 from RAM...")
    del text_encoder, prompt_hidden, prompt_ids, neg_hidden, neg_ids
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    log("STAGE 1 complete.")
    return embeds_path

# ---------------------------------------------------------------------------
#  Stage 2: Denoising with custom Anima transformer
# ---------------------------------------------------------------------------
def stage2_denoise(embeds_path, args, device, dtype):
    """Stage 2 - Run the custom Anima transformer denoising loop."""
    banner("STAGE 2 / 3  ::  Diffusion Transformer (Anima)")

    # 1. Download weights to a known path
    weights_path = hf_get(ANIMA_REPO, DIFFUSION_WEIGHT_FILE)

    # 2. Load custom model (matches Anima weight naming exactly)
    log("Loading Anima transformer (this takes ~60s, ~4GB RAM)...")
    transformer = load_anima_transformer(weights_path, dtype=dtype)
    log(f"  transformer params: {sum(p.numel() for p in transformer.parameters())/1e9:.2f}B")

    # 3. Set up flow-match scheduler manually (avoid diffusers dynamic_shifting issues)
    # Anima uses FlowMatchEulerDiscrete. For turbo (8 steps, CFG=1), use simple linear schedule.
    num_train_timesteps = 1000
    shift = 3.0
    sigmas = torch.linspace(1, 0, args.steps + 1, device=device)
    # Apply shift (Cosmos uses shift=3 for image)
    sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
    timesteps = (sigmas[:-1] * num_train_timesteps).long()
    log(f"  timesteps: {timesteps.tolist()}")

    # 4. Latents - 16-channel VAE latents
    latent_channels = 16
    latent_h = args.height // 8
    latent_w = args.width // 8

    if args.seed >= 0:
        torch.manual_seed(args.seed)
        log(f"  seed: {args.seed}")

    latents = torch.randn(
        (1, latent_channels, latent_h, latent_w),
        device=device,
        dtype=dtype,
    )
    log(f"  latents shape: {latents.shape}")

    # 5. Load prompt embeds
    embeds_data = torch.load(embeds_path, map_location=device, weights_only=True)
    prompt_hidden   = embeds_data["prompt_hidden"].to(device=device, dtype=dtype)
    prompt_ids      = embeds_data["prompt_ids"].to(device=device)
    negative_hidden = embeds_data["negative_hidden"].to(device=device, dtype=dtype)
    negative_ids    = embeds_data["negative_ids"].to(device=device)

    use_cfg = args.cfg > 1.0 + 1e-6
    log(f"  CFG: {args.cfg} ({'on' if use_cfg else 'off — turbo mode'})")

    log("Running denoising loop...")
    start = time.time()

    with torch.no_grad():
        for i, t in enumerate(timesteps):
            t_batch = torch.tensor([float(t)], device=device, dtype=dtype)

            if use_cfg:
                # Run both conditional and unconditional
                lat_in = torch.cat([latents, latents], dim=0)
                tok_in = torch.cat([negative_ids, prompt_ids], dim=0)
                hid_in = torch.cat([negative_hidden, prompt_hidden], dim=0)
                t_in = torch.cat([t_batch, t_batch], dim=0)

                noise_pred = transformer(lat_in, t_in, tok_in, hid_in)
                noise_uncond, noise_cond = noise_pred.chunk(2)
                noise_pred = noise_uncond + args.cfg * (noise_cond - noise_uncond)
            else:
                # Turbo mode — just conditional
                noise_pred = transformer(latents, t_batch, prompt_ids, prompt_hidden)

            # Flow-match Euler step: x_{t-1} = x_t + (t_{t-1} - t_t) * v
            # where v is the velocity prediction (= noise_pred for flow match)
            sigma_cur = sigmas[i]
            sigma_next = sigmas[i + 1]
            latents = latents + (sigma_next - sigma_cur) * noise_pred

            elapsed = time.time() - start
            total_est = elapsed / (i + 1) * len(timesteps)
            log(f"  step {i+1}/{len(timesteps)}  "
                f"elapsed={elapsed:.0f}s  eta={total_est-elapsed:.0f}s")

    total = time.time() - start
    log(f"Denoising complete in {total:.1f}s ({total/60:.1f} min)")

    latents_path = "/tmp/anima_latents.pt"
    torch.save(latents.cpu(), latents_path)

    log("Freeing transformer from RAM...")
    del transformer, latents, prompt_hidden, prompt_ids
    del negative_hidden, negative_ids, embeds_data
    if 'noise_pred' in dir(): del noise_pred
    if use_cfg and 'noise_uncond' in dir():
        del noise_uncond, noise_cond, lat_in, tok_in, hid_in, t_in
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    log("STAGE 2 complete.")
    return latents_path

# ---------------------------------------------------------------------------
#  Stage 3: VAE decode
# ---------------------------------------------------------------------------
def stage3_decode(latents_path, output_path, device, dtype):
    """Stage 3 - VAE Decode to image."""
    banner("STAGE 3 / 3  ::  VAE Decoder (Qwen-Image)")
    try:
        from diffusers import AutoencoderKLQwenImage as VAEClass
        log("Using AutoencoderKLQwenImage")
    except ImportError:
        from diffusers import AutoencoderKL as VAEClass
        log("AutoencoderKLQwenImage not available, falling back to AutoencoderKL")

    local_dir = build_local_component("vae", [
        {"repo": ANIMA_REPO,
         "file": VAE_WEIGHT_FILE,
         "rename_as": "diffusion_pytorch_model.safetensors"},
        {"local_path": "configs/vae/config.json"},
    ])

    log("Loading VAE...")
    vae = VAEClass.from_pretrained(local_dir, torch_dtype=dtype)
    vae.eval()
    log(f"  vae params: {sum(p.numel() for p in vae.parameters())/1e6:.1f}M")

    latents = torch.load(latents_path, map_location=device, weights_only=True)
    latents = latents.to(device=device, dtype=dtype)
    log(f"  latents shape: {latents.shape}")

    # Qwen-Image VAE uses latents_mean/std normalization
    # Get from config if available
    latents_mean = getattr(vae.config, "latents_mean", None)
    latents_std  = getattr(vae.config, "latents_std", None)
    if latents_mean is not None and latents_std is not None:
        latents_mean = torch.tensor(latents_mean, device=device, dtype=dtype).view(1, -1, 1, 1)
        latents_std  = torch.tensor(latents_std,  device=device, dtype=dtype).view(1, -1, 1, 1)
        latents = (latents - latents_mean) / latents_std
        log("  applied latents_mean/std normalization")

    with torch.no_grad():
        image = vae.decode(latents, return_dict=False)[0]

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).round().astype("uint8")

    from PIL import Image as PILImage
    pil = PILImage.fromarray(image[0])
    pil.save(output_path, format="PNG", optimize=True)
    log(f"  saved: {output_path} ({pil.size[0]}x{pil.size[1]})")

    del vae, latents, image
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    log("STAGE 3 complete.")
    return output_path

# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Anima T2I Generator (CPU-only)")
    parser.add_argument("--prompt",          required=True)
    parser.add_argument("--negative-prompt", default="worst quality, low quality")
    parser.add_argument("--output",          default="output.png")
    parser.add_argument("--chat-id",         default="")
    parser.add_argument("--width",           type=int,   default=1024)
    parser.add_argument("--height",          type=int,   default=1024)
    parser.add_argument("--steps",           type=int,   default=8)
    parser.add_argument("--cfg",             type=float, default=1.0)
    parser.add_argument("--seed",            type=int,   default=-1)
    args = parser.parse_args()

    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id   = args.chat_id or os.environ.get("TELEGRAM_CHAT_ID", "")

    banner("Anima T2I Runner  ::  CPU-Only Mode")
    log(f"Prompt : {args.prompt}")
    log(f"Size   : {args.width}x{args.height}")
    log(f"Steps  : {args.steps}  |  CFG: {args.cfg}")
    log(f"Seed   : {'random' if args.seed < 0 else args.seed}")

    tg_send_message(
        bot_token, chat_id,
        f"🎨 *Starting generation*\n\n"
        f"`{args.prompt[:200]}`\n\n"
        f"📐 {args.width}x{args.height} | {args.steps} steps | CFG {args.cfg}\n"
        f"⏱ CPU mode — ETA ~5-10 min (turbo)"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.float32  # CPU mode
    log(f"Device: {device}  |  dtype: {dtype}")

    # Set the global weight_dtype_next for timestep_embedding
    import anima_model
    anima_model.weight_dtype_next = dtype

    start = time.time()
    try:
        embeds_path  = stage1_encode_prompt(args.prompt, args.negative_prompt, device, dtype)
        latents_path = stage2_denoise(embeds_path, args, device, dtype)
        output_path  = stage3_decode(latents_path, args.output, device, dtype)

        total = time.time() - start
        log(f"\n✅ Generation complete in {total:.0f}s ({total/60:.1f} min)")

        if bot_token and chat_id:
            log("Sending photo to Telegram...")
            ok = tg_send_photo(
                bot_token, chat_id, output_path,
                caption=f"✅ `{args.prompt[:80]}`\n⏱ {total:.0f}s | {args.width}x{args.height}"
            )
            log("✅ Photo sent" if ok else "❌ Photo send failed")

        for f in [embeds_path, latents_path]:
            try: os.remove(f)
            except OSError: pass

        log(f"Final output: {output_path}")

    except Exception as e:
        log(f"\n❌ FATAL ERROR: {e}")
        traceback.print_exc()
        tg_send_message(
            bot_token, chat_id,
            f"❌ *Generation failed*\n\n`{str(e)[:300]}`"
        )
        sys.exit(1)

if __name__ == "__main__":
    main()
