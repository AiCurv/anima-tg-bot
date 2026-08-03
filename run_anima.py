#!/usr/bin/env python3
"""
================================================================================
 telegram-anima-diffusion-bot  ::  run_anima.py
================================================================================
 Pure-Python Anima (NVIDIA Cosmos-Predict2-2B-Text2Image) inference runner.

 Architecture:
   - Text Encoder: Qwen3 0.6B base (model.safetensors from Anima repo)
   - Transformer:  Anima (Cosmos2Transformer2DModel architecture)
   - VAE:          Qwen-Image VAE

 Memory staging (CRITICAL for 7GB RAM CPU runner):
   Stage 1 : Load Text Encoder  -> encode prompt  -> free
   Stage 2 : Load Transformer   -> denoise        -> free
   Stage 3 : Load VAE           -> decode         -> save image

 NO ComfyUI. NO Docker. NO GPU. Pure Python + diffusers.
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
#  Hugging Face download helpers
# ---------------------------------------------------------------------------
HF_TOKEN = os.environ.get("HF_TOKEN", None)

def hf_get(repo_id, filename):
    """Download a single file from HF Hub."""
    from huggingface_hub import hf_hub_download
    log(f"  -> hf: {repo_id} / {filename}")
    return hf_hub_download(repo_id=repo_id, filename=filename, token=HF_TOKEN)

def build_local_component(name, spec):
    """
    Build a local directory containing all files needed for from_pretrained().
    `spec` is a list of dicts:
      - {repo, file, rename_as}      -- download from HF Hub
      - {local_path, rename_as}      -- copy from local file in repo
    """
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
ANIMA_REPO          = "circlestone-labs/Anima"
NVIDIA_COSMOS_REPO  = "nvidia/Cosmos-Predict2-2B-Text2Image"
QWEN3_BASE_REPO     = "Qwen/Qwen3-0.6B-Base"
QWEN_IMAGE_REPO     = "Qwen/Qwen-Image"

# Default to turbo (8 steps, CFG 1) — much faster on CPU
DIFFUSION_WEIGHT_FILE = "split_files/diffusion_models/anima-turbo-v1.0.safetensors"
ENCODER_WEIGHT_FILE   = "split_files/text_encoders/qwen_3_06b_base.safetensors"
VAE_WEIGHT_FILE       = "split_files/vae/qwen_image_vae.safetensors"

# ---------------------------------------------------------------------------
#  Stage 1: Text Encoder
# ---------------------------------------------------------------------------
def stage1_encode_prompt(prompt, negative_prompt, device, dtype):
    """Stage 1 - Text Encoding with Qwen3 0.6B."""
    banner("STAGE 1 / 3  ::  Text Encoder (Qwen3 0.6B)")

    from transformers import AutoTokenizer, AutoModelForCausalLM

    # Anima repo only ships weights; configs come from local repo files.
    # Weights must be named `model.safetensors` for transformers to find them.
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

    log("Loading text encoder model...")
    # CPU-only: skip low_cpu_mem_usage (meta tensors can't be moved on CPU)
    text_encoder = AutoModelForCausalLM.from_pretrained(
        local_dir,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    text_encoder.eval()
    log(f"  text_encoder params: {sum(p.numel() for p in text_encoder.parameters())/1e6:.1f}M")

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
            # Use last hidden state as prompt embeddings
            return out.hidden_states[-1], inputs

    log(f"Encoding prompt: {prompt[:80]}")
    prompt_embeds, pos_inputs = encode(prompt)

    if negative_prompt:
        negative_embeds, _ = encode(negative_prompt)
    else:
        # Use empty string for unconditional (CFG=1 will make this irrelevant)
        negative_embeds, _ = encode("")

    log(f"  prompt_embeds shape: {prompt_embeds.shape}")

    # Save embeddings to disk so we can free the encoder
    embeds_path = "/tmp/anima_prompt_embeds.pt"
    torch.save({
        "prompt_embeds":   prompt_embeds.cpu(),
        "negative_embeds": negative_embeds.cpu(),
    }, embeds_path)

    # CLEANUP
    log("Freeing text encoder from RAM...")
    del text_encoder, prompt_embeds, negative_embeds, pos_inputs
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    log("STAGE 1 complete.")
    return embeds_path

# ---------------------------------------------------------------------------
#  Stage 2: Transformer (denoising)
# ---------------------------------------------------------------------------
def stage2_denoise(embeds_path, args, device, dtype):
    """Stage 2 - Diffusion Transformer denoising."""
    banner("STAGE 2 / 3  ::  Diffusion Transformer (Anima)")
    from diffusers import CosmosTransformer3DModel, FlowMatchEulerDiscreteScheduler

    # Anima repo only ships weights; config comes from local repo file.
    # Diffusers expects `diffusion_pytorch_model.safetensors`.
    local_dir = build_local_component("transformer", [
        {"repo": ANIMA_REPO,
         "file": DIFFUSION_WEIGHT_FILE,
         "rename_as": "diffusion_pytorch_model.safetensors"},
        {"local_path": "configs/transformer/config.json"},
    ])

    log("Loading transformer (this takes ~60s, ~4GB RAM)...")
    # NOTE: low_cpu_mem_usage=True creates meta tensors which can't be moved
    # with .to(device) on CPU. Since we're CPU-only, we skip low_cpu_mem_usage
    # and skip the explicit .to(device) call — model loads directly to CPU.
    transformer = CosmosTransformer3DModel.from_pretrained(
        local_dir,
        torch_dtype=dtype,
    )
    transformer.eval()
    # Already on CPU (default), no need to call .to(device)
    log(f"  transformer params: {sum(p.numel() for p in transformer.parameters())/1e9:.2f}B")

    # Scheduler
    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000,
        shift=3.0,
        use_dynamic_shifting=True,
        base_image_seq_len=256,
        max_image_seq_len=4096,
    )
    scheduler.set_timesteps(args.steps, device=device)
    log(f"  scheduler timesteps: {len(scheduler.timesteps)}")

    # Latents - Cosmos2 uses 16-channel VAE latents
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

    # Load prompt embeds
    embeds_data = torch.load(embeds_path, map_location=device, weights_only=True)
    prompt_embeds   = embeds_data["prompt_embeds"].to(device=device, dtype=dtype)
    negative_embeds = embeds_data["negative_embeds"].to(device=device, dtype=dtype)

    use_cfg = args.cfg > 1.0 + 1e-6
    log(f"  CFG: {args.cfg} ({'on' if use_cfg else 'off — turbo mode'})")

    log("Running denoising loop...")
    start = time.time()

    with torch.no_grad():
        for i, t in enumerate(scheduler.timesteps):
            t_input = t.unsqueeze(0).to(device)

            if use_cfg:
                latent_input = torch.cat([latents, latents], dim=0)
                embed_input  = torch.cat([negative_embeds, prompt_embeds], dim=0)
            else:
                latent_input = latents
                embed_input  = prompt_embeds

            try:
                out = transformer(
                    latent_input,
                    t_input,
                    encoder_hidden_states=embed_input,
                    return_dict=False,
                )
                noise_pred = out[0] if isinstance(out, tuple) else out.sample
            except Exception as e:
                log(f"  ! step {i} transformer call failed: {e}")
                raise

            if use_cfg:
                noise_uncond, noise_cond = noise_pred.chunk(2)
                noise_pred = noise_uncond + args.cfg * (noise_cond - noise_uncond)

            latents = scheduler.step(noise_pred, t, latents).prev_sample

            if (i + 1) % 2 == 0 or i == 0:
                elapsed = time.time() - start
                total_est = elapsed / (i + 1) * len(scheduler.timesteps)
                log(f"  step {i+1}/{len(scheduler.timesteps)}  "
                    f"elapsed={elapsed:.0f}s  eta={total_est-elapsed:.0f}s")

    total = time.time() - start
    log(f"Denoising complete in {total:.1f}s ({total/60:.1f} min)")

    latents_path = "/tmp/anima_latents.pt"
    torch.save(latents.cpu(), latents_path)

    # CLEANUP
    log("Freeing transformer from RAM...")
    del transformer, latents, prompt_embeds, negative_embeds, embeds_data
    try:
        del noise_pred
    except NameError:
        pass
    if use_cfg:
        try:
            del noise_uncond, noise_cond, latent_input, embed_input
        except NameError:
            pass
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
    vae = VAEClass.from_pretrained(
        local_dir,
        torch_dtype=dtype,
    )
    vae.eval()
    log(f"  vae params: {sum(p.numel() for p in vae.parameters())/1e6:.1f}M")

    latents = torch.load(latents_path, map_location=device, weights_only=True)
    latents = latents.to(device=device, dtype=dtype)
    log(f"  latents shape: {latents.shape}")

    scaling_factor = getattr(vae.config, "scaling_factor", 0.13025)
    log(f"  vae scaling_factor: {scaling_factor}")

    with torch.no_grad():
        image = vae.decode(latents / scaling_factor, return_dict=False)[0]

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
