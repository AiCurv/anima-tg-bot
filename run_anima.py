#!/usr/bin/env python3
"""
================================================================================
 telegram-anima-diffusion-bot  ::  run_anima.py
================================================================================
 Pure-Python Anima (NVIDIA Cosmos-Predict2-2B-Text2Image) inference runner.

 CRITICAL DESIGN:
   - NO ComfyUI, NO Docker, NO external GUI
   - CPU-only (GitHub Actions ubuntu-latest, 7 GB RAM)
   - Memory-staged execution:
       Stage 1 : Load Text Encoder  -> encode prompt  -> free
       Stage 2 : Load Transformer   -> denoise        -> free
       Stage 3 : Load VAE           -> decode         -> save image
   - Anonymous HF downloads (no token needed for public models)
   - Output is sent straight back to Telegram via Bot API
================================================================================
"""

import os
import sys
import gc
import time
import json
import shutil
import argparse
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
#  Banner
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
#  Hugging Face download helpers (anonymous - no token needed)
# ---------------------------------------------------------------------------
def hf_get(repo_id, filename, token=None):
    """Download a single file from HF Hub (anonymous if no token)."""
    from huggingface_hub import hf_hub_download
    log(f"  -> hf: {repo_id} / {filename}")
    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        token=token,
    )

def build_local_component(name, files):
    """
    Build a local directory containing all files needed for from_pretrained().
    `files` is a list of (repo_id, filename) tuples.
    Returns the path to the local directory.
    """
    local_dir = Path(f"/tmp/anima_components/{name}")
    local_dir.mkdir(parents=True, exist_ok=True)
    for repo_id, filename in files:
        src = hf_get(repo_id, filename)
        dst = local_dir / Path(filename).name
        if not dst.exists():
            shutil.copy(src, dst)
            log(f"  -> staged: {dst}")
    return str(local_dir)

# ---------------------------------------------------------------------------
#  Pipeline configuration
# ---------------------------------------------------------------------------
ANIMA_REPO = "circlestone-labs/Anima"

# Diffusion weights
DIFFUSION_WEIGHT_FILE = "split_files/diffusion_models/anima-base-v1.0.safetensors"

# Text encoder weights
ENCODER_WEIGHT_FILE = "split_files/text_encoders/qwen_3_06b_base.safetensors"

# VAE weights
VAE_WEIGHT_FILE = "split_files/vae/qwen_image_vae.safetensors"

# Config fallback sources (if Anima repo doesn't ship configs in subfolders)
# We try Anima repo first, then fall back to NVIDIA's Cosmos-Predict2 reference.
NVIDIA_COSMOS_REPO = "nvidia/Cosmos-Predict2-2B-Text2Image"
QWEN3_BASE_REPO   = "Qwen/Qwen3-0.6B-Base"
QWEN_IMAGE_VAE_REPO = "Qwen/Qwen-Image"  # Qwen-Image ships the same VAE class

# ---------------------------------------------------------------------------
#  Stage 1: Text Encoder
# ---------------------------------------------------------------------------
def stage1_encode_prompt(prompt, negative_prompt, device, dtype):
    """
    Stage 1 - Text Encoding
    Downloads and loads the Qwen3 0.6B text encoder, tokenizes the prompt,
    produces prompt embeddings, then frees all encoder memory before returning.
    """
    banner("STAGE 1 / 3  ::  Text Encoder (Qwen3 0.6B)")

    from transformers import AutoTokenizer, AutoModelForCausalLM

    # 1. Build local dir with weights + config (try multiple config sources)
    encoder_files = [(ANIMA_REPO, ENCODER_WEIGHT_FILE)]

    # Try Anima repo config first
    try:
        encoder_files.append((ANIMA_REPO, "split_files/text_encoders/config.json"))
        local_dir = build_local_component("text_encoder", encoder_files)
        # Verify config.json exists in local dir
        if not (Path(local_dir) / "config.json").exists():
            raise FileNotFoundError
    except Exception:
        log("  ! Anima text_encoder config missing, falling back to Qwen3-0.6B-Base")
        encoder_files = [
            (ANIMA_REPO, ENCODER_WEIGHT_FILE),
            (QWEN3_BASE_REPO, "config.json"),
            (QWEN3_BASE_REPO, "tokenizer_config.json"),
            (QWEN3_BASE_REPO, "tokenizer.json"),
            (QWEN3_BASE_REPO, "vocab.json"),
            (QWEN3_BASE_REPO, "merges.txt"),
        ]
        local_dir = build_local_component("text_encoder", encoder_files)

    log("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(local_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log("Loading text encoder model (this takes ~30s)...")
    text_encoder = AutoModelForCausalLM.from_pretrained(
        local_dir,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    text_encoder.eval()
    text_encoder.to(device)
    log(f"  text_encoder memory: {sum(p.numel() for p in text_encoder.parameters())/1e6:.1f}M params")

    # 2. Tokenize prompt with a chat-style template (Qwen3 expects this)
    def format_for_encoder(text):
        # Simple text-only format - Qwen3 base model accepts raw text
        return text

    log(f"Encoding prompt: {prompt[:80]}")
    pos_inputs = tokenizer(
        format_for_encoder(prompt),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
    ).to(device)

    with torch.no_grad():
        pos_outputs = text_encoder(**pos_inputs, output_hidden_states=True)
        # Use last hidden state as prompt embeddings
        prompt_embeds = pos_outputs.hidden_states[-1]

    # Negative prompt (for CFG)
    if negative_prompt:
        neg_inputs = tokenizer(
            format_for_encoder(negative_prompt),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        ).to(device)
        with torch.no_grad():
            neg_outputs = text_encoder(**neg_inputs, output_hidden_states=True)
            negative_embeds = neg_outputs.hidden_states[-1]
    else:
        # Use empty / zero embeddings for unconditional
        negative_embeds = torch.zeros_like(prompt_embeds)

    log(f"  prompt_embeds shape: {prompt_embeds.shape}")

    # 3. Save embeddings to disk so we can free the encoder fully
    embeds_path = "/tmp/anima_prompt_embeds.pt"
    torch.save({"prompt_embeds": prompt_embeds.cpu(),
                "negative_embeds": negative_embeds.cpu()}, embeds_path)

    # 4. CLEANUP - critical for 7GB RAM runner
    log("Freeing text encoder from RAM...")
    del text_encoder, pos_outputs, neg_outputs, pos_inputs, neg_inputs
    del prompt_embeds, negative_embeds
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    log("STAGE 1 complete.")
    return embeds_path, tokenizer

# ---------------------------------------------------------------------------
#  Stage 2: Transformer (denoising)
# ---------------------------------------------------------------------------
def stage2_denoise(embeds_path, args, device, dtype):
    """
    Stage 2 - Diffusion Transformer
    Loads the Anima transformer, runs the denoising loop, produces latents.
    Frees transformer before returning.
    """
    banner("STAGE 2 / 3  ::  Diffusion Transformer (Anima base v1.0)")

    # Try to import Cosmos2 transformer class (diffusers >= 0.32)
    try:
        from diffusers import Cosmos2Transformer2DModel, FlowMatchEulerDiscreteScheduler
        log("Using Cosmos2Transformer2DModel from diffusers")
    except ImportError:
        log("ERROR: diffusers too old. Need >=0.32 for Cosmos2 support.")
        log("Falling back to generic transformer loading...")
        from diffusers import Transformer2DModel as Cosmos2Transformer2DModel
        from diffusers import FlowMatchEulerDiscreteScheduler

    # 1. Build local dir with weights + config
    transformer_files = [(ANIMA_REPO, DIFFUSION_WEIGHT_FILE)]
    try:
        transformer_files.append((ANIMA_REPO, "split_files/diffusion_models/config.json"))
        local_dir = build_local_component("transformer", transformer_files)
        if not (Path(local_dir) / "config.json").exists():
            raise FileNotFoundError
    except Exception:
        log("  ! Anima transformer config missing, trying NVIDIA Cosmos-Predict2 config")
        try:
            transformer_files = [
                (ANIMA_REPO, DIFFUSION_WEIGHT_FILE),
                (NVIDIA_COSMOS_REPO, "transformer/config.json"),
            ]
            local_dir = build_local_component("transformer", transformer_files)
        except Exception as e:
            log(f"  ! NVIDIA fallback also failed: {e}")
            log("  ! Trying Anima repo config from root...")
            transformer_files = [
                (ANIMA_REPO, DIFFUSION_WEIGHT_FILE),
                (ANIMA_REPO, "config.json"),
            ]
            local_dir = build_local_component("transformer", transformer_files)

    log("Loading transformer (this takes ~60s, ~2GB RAM)...")
    try:
        transformer = Cosmos2Transformer2DModel.from_pretrained(
            local_dir,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
    except Exception as e:
        log(f"  ! from_pretrained failed: {e}")
        log("  ! Attempting manual state_dict load...")
        from safetensors.torch import load_file
        weight_path = hf_get(ANIMA_REPO, DIFFUSION_WEIGHT_FILE)
        state_dict = load_file(weight_path)
        # Try to find a config.json we already staged
        config_path = Path(local_dir) / "config.json"
        if config_path.exists():
            import json as _json
            config = _json.loads(config_path.read_text())
            transformer = Cosmos2Transformer2DModel(**config)
            transformer.load_state_dict(state_dict, strict=False)
        else:
            raise RuntimeError("Cannot load transformer without config")

    transformer.eval()
    transformer.to(device)
    log(f"  transformer memory: {sum(p.numel() for p in transformer.parameters())/1e9:.2f}B params")

    # 2. Set up scheduler (Flow Match Euler - standard for Cosmos2)
    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000,
        shift=3.0,
        use_dynamic_shifting=True,
        base_image_seq_len=256,
        max_image_seq_len=4096,
    )
    scheduler.set_timesteps(args.steps, device=device)
    log(f"  scheduler timesteps: {len(scheduler.timesteps)}")

    # 3. Prepare latents
    # Cosmos2 uses 16-channel VAE latents (typical for high-res T2I)
    latent_channels = 16
    latent_h = args.height // 8
    latent_w = args.width // 8

    if args.seed >= 0:
        torch.manual_seed(args.seed)
        log(f"  using seed: {args.seed}")

    latents = torch.randn(
        (1, latent_channels, latent_h, latent_w),
        device=device,
        dtype=dtype,
    )
    log(f"  latents shape: {latents.shape}")

    # 4. Load prompt embeds
    embeds_data = torch.load(embeds_path, map_location=device)
    prompt_embeds = embeds_data["prompt_embeds"].to(device=device, dtype=dtype)
    negative_embeds = embeds_data["negative_embeds"].to(device=device, dtype=dtype)

    # 5. Denoising loop with CFG
    use_cfg = args.cfg > 1.0
    if use_cfg:
        log(f"  CFG enabled: {args.cfg}")
    else:
        log("  CFG disabled (cfg=1)")

    log("Running denoising loop...")
    start_time = time.time()

    with torch.no_grad():
        for i, t in enumerate(scheduler.timesteps):
            t_input = t.unsqueeze(0).to(device)

            if use_cfg:
                latent_input = torch.cat([latents, latents], dim=0)
                embed_input = torch.cat([negative_embeds, prompt_embeds], dim=0)
            else:
                latent_input = latents
                embed_input = prompt_embeds

            try:
                noise_pred = transformer(
                    latent_input,
                    t_input,
                    encoder_hidden_states=embed_input,
                    return_dict=False,
                )
                if isinstance(noise_pred, tuple):
                    noise_pred = noise_pred[0]
            except Exception as e:
                log(f"  ! transformer step {i} failed: {e}")
                # Try alternative calling convention
                noise_pred = transformer(
                    latent_input,
                    t_input,
                    encoder_hidden_states=embed_input,
                ).sample

            if use_cfg:
                noise_uncond, noise_cond = noise_pred.chunk(2)
                noise_pred = noise_uncond + args.cfg * (noise_cond - noise_uncond)

            latents = scheduler.step(noise_pred, t, latents).prev_sample

            if (i + 1) % 5 == 0 or i == 0:
                elapsed = time.time() - start_time
                est_total = elapsed / (i + 1) * len(scheduler.timesteps)
                log(f"  step {i+1}/{len(scheduler.timesteps)}  "
                    f"elapsed={elapsed:.0f}s  eta={est_total-elapsed:.0f}s")

    total_time = time.time() - start_time
    log(f"Denoising complete in {total_time:.1f}s")

    # 6. Save latents
    latents_path = "/tmp/anima_latents.pt"
    torch.save(latents.cpu(), latents_path)

    # 7. CLEANUP
    log("Freeing transformer from RAM...")
    del transformer, latents, prompt_embeds, negative_embeds, embeds_data
    # These are only defined if the denoising loop ran at least once
    for _name in ("noise_pred", "noise_uncond", "noise_cond", "latent_input", "embed_input"):
        if _name in dir():
            try:
                exec(f"del {_name}")
            except Exception:
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
    """
    Stage 3 - VAE Decoder
    Loads the Qwen-Image VAE, decodes latents to pixel image, saves PNG.
    """
    banner("STAGE 3 / 3  ::  VAE Decoder (Qwen-Image)")

    from diffusers import AutoencoderKL

    # 1. Build local dir
    vae_files = [(ANIMA_REPO, VAE_WEIGHT_FILE)]
    try:
        vae_files.append((ANIMA_REPO, "split_files/vae/config.json"))
        local_dir = build_local_component("vae", vae_files)
        if not (Path(local_dir) / "config.json").exists():
            raise FileNotFoundError
    except Exception:
        log("  ! Anima VAE config missing, trying Qwen-Image repo")
        vae_files = [
            (ANIMA_REPO, VAE_WEIGHT_FILE),
            (QWEN_IMAGE_VAE_REPO, "vae/config.json"),
        ]
        try:
            local_dir = build_local_component("vae", vae_files)
        except Exception:
            # Final fallback - try common VAE config pattern
            vae_files = [
                (ANIMA_REPO, VAE_WEIGHT_FILE),
                (NVIDIA_COSMOS_REPO, "vae/config.json"),
            ]
            local_dir = build_local_component("vae", vae_files)

    log("Loading VAE (this takes ~15s, ~500MB RAM)...")
    vae = AutoencoderKL.from_pretrained(
        local_dir,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    vae.eval()
    vae.to(device)
    log(f"  vae memory: {sum(p.numel() for p in vae.parameters())/1e6:.1f}M params")

    # 2. Load latents
    latents = torch.load(latents_path, map_location=device)
    latents = latents.to(device=device, dtype=dtype)
    log(f"  latents shape: {latents.shape}")

    # 3. Decode
    scaling_factor = getattr(vae.config, "scaling_factor", 0.13025)
    log(f"  vae scaling_factor: {scaling_factor}")

    with torch.no_grad():
        latents_scaled = latents / scaling_factor
        image = vae.decode(latents_scaled, return_dict=False)[0]

    # 4. Post-process to PIL Image
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).round().astype("uint8")

    from PIL import Image as PILImage
    pil_image = PILImage.fromarray(image[0])
    pil_image.save(output_path, format="PNG", optimize=True)
    log(f"  saved image: {output_path}  ({pil_image.size[0]}x{pil_image.size[1]})")

    # 5. CLEANUP
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
    parser.add_argument("--prompt", required=True, help="Text prompt")
    parser.add_argument("--negative-prompt", default="", help="Negative prompt")
    parser.add_argument("--output", default="output.png", help="Output image path")
    parser.add_argument("--chat-id", default="", help="Telegram chat ID for status updates")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--cfg", type=float, default=4.5, help="CFG scale (4-5 for base)")
    parser.add_argument("--seed", type=int, default=-1, help="-1 for random")
    args = parser.parse_args()

    # Resolve secrets
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id   = args.chat_id or os.environ.get("TELEGRAM_CHAT_ID", "")
    hf_token  = os.environ.get("HF_TOKEN", "")  # optional - anonymous works for public

    banner(f"Anima T2I Runner  ::  CPU-Only Mode")
    log(f"Prompt : {args.prompt}")
    log(f"Size   : {args.width}x{args.height}")
    log(f"Steps  : {args.steps}  |  CFG: {args.cfg}")
    log(f"Seed   : {'random' if args.seed < 0 else args.seed}")

    # Notify Telegram: starting
    tg_send_message(
        bot_token, chat_id,
        f"🎨 *Starting generation*\n\n"
        f"`{args.prompt[:200]}`\n\n"
        f"📐 {args.width}x{args.height} | {args.steps} steps | CFG {args.cfg}\n"
        f"⏱ CPU mode - this takes ~10-15 min"
    )

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.float32  # CPU needs float32 (bf16/fp16 unsupported on most CPUs)
    log(f"Device: {device}  |  dtype: {dtype}")

    start = time.time()

    try:
        # ---- STAGE 1 ----
        embeds_path, _tokenizer = stage1_encode_prompt(
            args.prompt, args.negative_prompt, device, dtype
        )

        # ---- STAGE 2 ----
        latents_path = stage2_denoise(embeds_path, args, device, dtype)

        # ---- STAGE 3 ----
        output_path = stage3_decode(latents_path, args.output, device, dtype)

        total_time = time.time() - start
        log(f"\n✅ Generation complete in {total_time:.0f}s ({total_time/60:.1f} min)")

        # Send photo to Telegram
        if bot_token and chat_id:
            log("Sending photo to Telegram...")
            ok = tg_send_photo(
                bot_token, chat_id, output_path,
                caption=f"✅ `{args.prompt[:100]}`\n⏱ {total_time:.0f}s | {args.width}x{args.height}"
            )
            if ok:
                log("✅ Photo sent successfully")
            else:
                log("❌ Photo send failed")
        else:
            log("(no Telegram creds - skipping send)")

        # Cleanup temp files
        for f in [embeds_path, latents_path]:
            try:
                os.remove(f)
            except Exception:
                pass

        log(f"\nFinal output: {output_path}")

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
