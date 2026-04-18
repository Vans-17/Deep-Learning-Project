# src/diffusion/ldm.py
"""
LDM loader.
Loads CompVis/stable-diffusion-v1-4 in fp16 and exposes the four objects
that the rest of the pipeline needs:
    vae       : AutoencoderKL
    unet      : UNet2DConditionModel
    scheduler : DDIMScheduler
    NULL_EMB  : Tensor (1, 77, 768) — null text conditioning

All model weights are frozen (requires_grad=False).
xformers memory-efficient attention is enabled if available.
"""

import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline

from ..config import cfg


def load_ldm(model_id: str = cfg.ldm_model_id):
    """
    Load the Stable Diffusion pipeline and extract its components.

    Args:
        model_id: HuggingFace model identifier.

    Returns:
        (vae, unet, scheduler, NULL_EMB) — all on cfg.device / cfg.dtype.
    """
    print(f"Loading LDM: {model_id} …")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=cfg.dtype,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(cfg.device)

    # Optional memory-efficient attention
    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("  xformers memory-efficient attention enabled ✅")
    except Exception:
        print("  xformers not available — using standard attention")

    vae      = pipe.vae.eval()
    unet     = pipe.unet.eval()
    scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    scheduler.set_timesteps(cfg.ddim_steps_denoise)

    # Freeze all weights — we never train the backbone
    for model in [vae, unet, pipe.text_encoder]:
        for p in model.parameters():
            p.requires_grad_(False)

    # Pre-compute null text embedding (used for CFG unconditional branch)
    with torch.no_grad():
        null_tok = pipe.tokenizer(
            [""],
            return_tensors="pt",
            padding="max_length",
            max_length=77,
            truncation=True,
        ).input_ids.to(cfg.device)
        NULL_EMB = pipe.text_encoder(null_tok)[0]  # (1, 77, 768)

    vram_used = (
        torch.cuda.memory_allocated() / 1e9 if cfg.device == "cuda" else 0.0
    )
    print(f"✅ LDM loaded | VRAM used: {vram_used:.2f} GB")
    print(f"   UNet params: {sum(p.numel() for p in unet.parameters()) / 1e6:.0f}M")

    return vae, unet, scheduler, NULL_EMB
