# src/pipeline.py
"""
Full LDFaceNet-Lite + BFM pipeline.

Exposes:
    init_models()        — load all models once; must be called first
    run_pipeline()       — single face swap call
    save_pipeline()      — checkpoint all conditioning modules
    load_pipeline()      — restore from checkpoint
    run_ablation()       — grid ablation over lambda configs
    run_presets()        — named preset comparison
    interactive_demo()   — ipywidgets slider UI (Jupyter only)
"""

import gc
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from diffusers import DDIMScheduler

from src.config import cfg
from src.extractors.arcface import ArcFaceIdentityExtractor
from src.extractors.bfm import BFMExtractor
from src.extractors.clip import CLIPDisentangler
from src.extractors.segmentor import LiteFaceSegmentor
from src.adaptors.identity_projector import IdentityProjector
from src.adaptors.illumination_adain import IlluminationAdaIN
from src.adaptors.pose_expr_adaptor import PoseExpressionAdapter
from src.diffusion.ldm import load_ldm
from src.diffusion.vae_helpers import encode_image, latent_blend_and_decode
from src.diffusion.denoiser import GuidedDenoiser
from src.metrics import compute_metrics


# ── Module-level model registry ───────────────────────────────────────────────
# Populated once by init_models(); reused by every run_pipeline() call.
_models: dict = {}


def init_models(model_id: str = cfg.ldm_model_id) -> dict:
    """
    Load and initialise all models. Call once before run_pipeline().

    Returns:
        dict of all model objects (also stored internally in _models).
    """
    global _models

    gc.collect()
    if cfg.device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    print("── Initialising models ──────────────────────────────────────────")

    arcface     = ArcFaceIdentityExtractor(device=cfg.device)
    tdmm        = BFMExtractor(device=cfg.device)
    bisenet     = LiteFaceSegmentor(device=cfg.device)
    clip_dis    = CLIPDisentangler(device=cfg.device)
    id_proj     = IdentityProjector(device=cfg.device)
    pose_adap   = PoseExpressionAdapter(device=cfg.device)
    illum_adain = IlluminationAdaIN(device=cfg.device)

    vae, unet, scheduler, NULL_EMB = load_ldm(model_id)

    denoiser = GuidedDenoiser(
        unet=unet,
        NULL_EMB=NULL_EMB,
        scheduler_config=scheduler.config,
        injection_scale=cfg.adapter_injection_scale,
    )

    _models = {
        "arcface"    : arcface,
        "tdmm"       : tdmm,
        "bisenet"    : bisenet,
        "clip_dis"   : clip_dis,
        "id_proj"    : id_proj,
        "pose_adap"  : pose_adap,
        "illum_adain": illum_adain,
        "vae"        : vae,
        "unet"       : unet,
        "scheduler"  : scheduler,
        "NULL_EMB"   : NULL_EMB,
        "denoiser"   : denoiser,
    }

    print("── All models ready ─────────────────────────────────────────────")
    return _models


# ── Pose safety clamp ─────────────────────────────────────────────────────────

def _safe_pose(pose_tensor: torch.Tensor) -> torch.Tensor:
    """Clamp yaw and pitch to prevent extreme-angle artefacts."""
    p = pose_tensor.clone()
    p[0] = p[0].clamp(-cfg.max_yaw_deg,   cfg.max_yaw_deg)
    p[1] = p[1].clamp(-cfg.max_pitch_deg, cfg.max_pitch_deg)
    return p


# ── Main pipeline ─────────────────────────────────────────────────────────────

@torch.no_grad()
def run_pipeline(
    source_img: np.ndarray,
    target_img: np.ndarray,
    lambda_id: float      = cfg.default_lambda_id,
    lambda_pose: float    = cfg.default_lambda_pose,
    lambda_shape: float   = cfg.default_lambda_shape,
    lambda_expr: float    = cfg.default_lambda_expr,
    lambda_light: float   = cfg.default_lambda_light,
    strength: float       = cfg.default_strength,
    num_steps: int        = cfg.ddim_steps_denoise,
    blend_hardness: float = cfg.default_blend_hardness,
    verbose: bool         = True,
) -> dict:
    """
    Run the full LDFaceNet-Lite + BFM face swap pipeline.

    DiffSwap convention:
        shape_coeff → from SOURCE  (identity geometry)
        pose, expr  → from TARGET  (animation)

    Args:
        source_img    : HxWx3 uint8 numpy — identity donor.
        target_img    : HxWx3 uint8 numpy — pose / expression donor.
        lambda_id     : identity token strength.
        lambda_pose   : pose residual injection strength.
        lambda_shape  : BFM shape coefficient strength (identity geometry).
        lambda_expr   : expression coefficient strength.
        lambda_light  : illumination AdaIN strength (wired, not yet hooked into UNet).
        strength      : img2img noise strength in (0, 1].
        num_steps     : DDIM denoising steps.
        blend_hardness: face mask edge sharpness.
        verbose       : print stage progress.

    Returns:
        dict with keys:
            result        : HxWx3 uint8 numpy — swapped face image
            identity_sim  : float | None — ArcFace cosine similarity
            attributes    : dict — tgt BFM attributes
            src_shape     : Tensor (1, 40) — source shape coefficients
            tgt_expr      : Tensor (1, 10) — target expression coefficients
    """
    if not _models:
        raise RuntimeError("Call init_models() before run_pipeline().")

    arcface   = _models["arcface"]
    tdmm      = _models["tdmm"]
    bisenet   = _models["bisenet"]
    clip_dis  = _models["clip_dis"]
    id_proj   = _models["id_proj"]
    pose_adap = _models["pose_adap"]
    vae       = _models["vae"]
    unet      = _models["unet"]
    scheduler = _models["scheduler"]
    NULL_EMB  = _models["NULL_EMB"]

    # ── Stage 1: attribute extraction ─────────────────────────────────────
    if verbose:
        print("━━ Stage 1: Attribute Extraction (BFM) ━━")

    src_id_emb = arcface.extract(source_img)
    src_bfm    = tdmm.extract(source_img)
    tgt_bfm    = tdmm.extract(target_img)

    for label, attrs in [("Source", src_bfm), ("Target", tgt_bfm)]:
        yaw = attrs["pose"][0].item()
        if abs(yaw) > cfg.max_yaw_deg:
            print(f"  ⚠️  {label} yaw={yaw:.1f}° > {cfg.max_yaw_deg}° — clamping")

    lmks_2d = tgt_bfm["landmarks"]
    _, tgt_face_mask, _ = bisenet.parse(target_img, lmks_2d)

    # DiffSwap mixing: SOURCE shape + TARGET pose & expression
    src_shape = src_bfm["shape_coeff"].unsqueeze(0)           # (1, 40)
    tgt_pose  = _safe_pose(tgt_bfm["pose"]).unsqueeze(0)      # (1,  6)
    tgt_expr  = tgt_bfm["expr_coeff"].unsqueeze(0)            # (1, 10)

    if verbose:
        print(f"  src shape norm : {src_shape.norm():.3f}")
        print(
            f"  tgt pose yaw   : {tgt_bfm['pose'][0].item():.1f}°"
            f"  →  clamped to {tgt_pose[0, 0].item():.1f}°"
        )
        print(f"  tgt expr norm  : {tgt_expr.norm():.3f}")
        print("  Stage 1 ✓")

    # ── Stage 2: CLIP disentangle + project identity tokens ───────────────
    if verbose:
        print("━━ Stage 2: CLIP Disentangle + Project ━━")

    src_clip  = clip_dis.extract(source_img)
    clip_id   = src_clip["id_feat"].to(cfg.device)
    id_tokens = id_proj(src_id_emb, clip_id, lambda_id=lambda_id)  # (1, 2, 768)

    pose_expr_out = pose_adap(
        pose_6d   = tgt_pose.to(cfg.device),
        shape_40d = src_shape.to(cfg.device),
        expr_10d  = tgt_expr.to(cfg.device),
        lambda_pose  = lambda_pose,
        lambda_shape = lambda_shape,
        lambda_expr  = lambda_expr,
    )

    if verbose:
        print("  Stage 2 ✓")

    # ── Stage 3: VAE encode target ─────────────────────────────────────────
    if verbose:
        print("━━ Stage 3: VAE Encode ━━")

    z_target = encode_image(target_img, vae)

    if verbose:
        print(f"  z_target: min={z_target.min():.3f} max={z_target.max():.3f}")

    # ── Stage 4: img2img guided denoising ─────────────────────────────────
    if verbose:
        print("━━ Stage 4: img2img Denoise ━━")

    den_sch = DDIMScheduler.from_config(scheduler.config)
    den_sch.set_timesteps(num_steps)

    start_step = int(num_steps * (1 - strength))
    timesteps  = den_sch.timesteps[start_step:]
    t_start    = timesteps[0]

    noise = torch.randn_like(z_target)
    a_t   = den_sch.alphas_cumprod[t_start].to(cfg.device)
    z     = a_t.sqrt() * z_target.float() + (1 - a_t).sqrt() * noise

    if verbose:
        print(f"  Starting at t={t_start.item()}, {len(timesteps)} steps")

    residuals = pose_expr_out["residuals"]

    for step_idx, t in enumerate(timesteps):
        if verbose and step_idx % 5 == 0:
            print(f"  Step {step_idx + 1}/{len(timesteps)}")

        # CFG: null + identity tokens concatenated as conditioning context
        cond_hs   = torch.cat(
            [NULL_EMB.expand(z.shape[0], -1, -1).to(cfg.dtype),
             id_tokens.to(cfg.dtype)],
            dim=1,
        )
        uncond_hs = torch.zeros_like(cond_hs)
        cfg_hs    = torch.cat([uncond_hs, cond_hs])

        latent_input   = torch.cat([z.half()] * 2)
        noise_pred_all = unet(latent_input, t, encoder_hidden_states=cfg_hs).sample
        noise_uncond, noise_cond = noise_pred_all.chunk(2)
        noise_pred = (noise_uncond + 3.5 * (noise_cond - noise_uncond)).float()

        # Scheduled residual injection — first 90 % of steps only
        if step_idx < int(len(timesteps) * 0.9) and residuals is not None:
            res = (
                residuals[0]
                .to(torch.float32)
                .mean(dim=1, keepdim=True)
                .expand_as(noise_pred)
            )
            noise_pred = noise_pred + (
                pose_adap.INJECTION_SCALE * lambda_pose
            ) * torch.clamp(res, -0.5, 0.5)

        z = den_sch.step(noise_pred, t, z).prev_sample.float()
        z = z.clamp(-8, 8)

    if verbose:
        print(f"  z_denoised: min={z.min():.3f} max={z.max():.3f}")

    # ── Stage 5: latent blend + decode ────────────────────────────────────
    if verbose:
        print("━━ Stage 5: Blend + Decode ━━")

    result_tensor = latent_blend_and_decode(
        z_swap        = z,
        z_target      = z_target,
        face_mask     = tgt_face_mask,
        vae           = vae,
        blend_hardness= blend_hardness,
    )

    result = (result_tensor.float().clamp(-1, 1) + 1) / 2
    result = (result[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

    # ── Post: identity similarity ──────────────────────────────────────────
    try:
        res_id = arcface.extract(result)
        id_sim = arcface.cosine_sim(src_id_emb, res_id)
        if verbose:
            print(f"  ID-sim (ArcFace cosine): {id_sim:.4f}")
    except Exception:
        id_sim = None

    if verbose:
        print("✅ Done")

    return {
        "result"      : result,
        "identity_sim": id_sim,
        "attributes"  : tgt_bfm,
        "src_shape"   : src_shape.cpu(),
        "tgt_expr"    : tgt_expr.cpu(),
    }


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_pipeline(path: Path = cfg.checkpoint_path):
    """Save state dicts of the four trainable conditioning modules."""
    if not _models:
        raise RuntimeError("Call init_models() first.")
    torch.save(
        {
            "id_proj"    : _models["id_proj"].state_dict(),
            "pose_adap"  : _models["pose_adap"].state_dict(),
            "illum_adain": _models["illum_adain"].state_dict(),
            "clip_dis"   : _models["clip_dis"].state_dict(),
        },
        path,
    )
    print(f"Saved → {path}")


def load_pipeline(path: Path = cfg.checkpoint_path):
    """Restore state dicts of the four conditioning modules from a checkpoint."""
    if not _models:
        raise RuntimeError("Call init_models() first.")
    ckpt = torch.load(path, map_location=cfg.device)
    _models["id_proj"].load_state_dict(ckpt["id_proj"])
    _models["pose_adap"].load_state_dict(ckpt["pose_adap"])
    _models["illum_adain"].load_state_dict(ckpt["illum_adain"])
    _models["clip_dis"].load_state_dict(ckpt["clip_dis"])
    print(f"Loaded ← {path}")


# ── Ablation ──────────────────────────────────────────────────────────────────

ABLATION_CONFIGS: dict = {
    "1. Baseline (Full)"   : {"id": 1.0, "pose": 1.0, "expr": 1.0, "shape": 1.0},
    "2. w/o Pose"          : {"id": 1.0, "pose": 0.0, "expr": 1.0, "shape": 1.0},
    "3. w/o Expression"    : {"id": 1.0, "pose": 1.0, "expr": 0.0, "shape": 1.0},
    "4. w/o Identity"      : {"id": 0.0, "pose": 1.0, "expr": 1.0, "shape": 1.0},
    "5. w/o Pose+Expr"     : {"id": 1.0, "pose": 0.0, "expr": 0.0, "shape": 1.0},
    "6. w/o Shape"         : {"id": 1.0, "pose": 1.0, "expr": 1.0, "shape": 0.0},
    "7. ID Only"           : {"id": 1.0, "pose": 0.0, "expr": 0.0, "shape": 0.0},
    "8. Shape+ID Only"     : {"id": 1.0, "pose": 0.0, "expr": 0.0, "shape": 1.0},
}


def run_ablation(
    src_img: np.ndarray,
    tgt_img: np.ndarray,
    configs: Optional[dict] = None,
    num_steps: int = 50,
    strength: float = 0.65,
    blend_hardness: float = 0.5,
    save_path: Optional[Path] = cfg.ablation_path,
) -> tuple:
    """Run all ablation configs and plot a comparison grid."""
    if configs is None:
        configs = ABLATION_CONFIGS

    arcface = _models["arcface"]
    results: dict = {}
    metrics: dict = {}

    for title, config in configs.items():
        print(f"\nRunning: {title} …")
        output = run_pipeline(
            source_img    = src_img,
            target_img    = tgt_img,
            lambda_id     = config["id"],
            lambda_pose   = config["pose"],
            lambda_expr   = config["expr"],
            lambda_shape  = config["shape"],
            lambda_light  = 0.5,
            strength      = strength,
            num_steps     = num_steps,
            blend_hardness= blend_hardness,
            verbose       = False,
        )
        results[title] = output["result"]
        metrics[title] = compute_metrics(src_img, tgt_img, output["result"], arcface)

    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    for ax, (title, img) in zip(axes, results.items()):
        m = metrics[title]
        ax.imshow(img)
        ax.set_title(
            f"{title}\nID={m['id_sim']:.3f}\nSSIM={m['ssim']:.3f}",
            fontsize=9, fontweight="bold",
        )
        ax.axis("off")
    plt.suptitle("Ablation Study — LDFaceNet-Lite + BFM", fontweight="bold", fontsize=13)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\nAblation grid saved → {save_path}")
    plt.show()

    print(f"\n{'Config':<25} {'ID-sim':>8} {'SSIM':>8} {'L2↓':>10}")
    print("─" * 55)
    for title, m in metrics.items():
        print(f"{title:<25} {m['id_sim']:>8.4f} {m['ssim']:>8.4f} {m['l2']:>10.1f}")

    return results, metrics


# ── Presets ───────────────────────────────────────────────────────────────────

def run_presets(
    src_img: np.ndarray,
    tgt_img: np.ndarray,
    save_path: Optional[Path] = Path("preset_comparison.png"),
) -> dict:
    """Run three named presets and plot side-by-side with source and target."""
    PRESETS = {
        "Pose only"      : dict(lambda_id=2.5, lambda_pose=1.5, lambda_expr=0.0, lambda_shape=0.0),
        "Expression only": dict(lambda_id=2.5, lambda_pose=0.0, lambda_expr=1.5, lambda_shape=0.0),
        "Fix identity"   : dict(lambda_id=4.0, lambda_pose=0.3, lambda_expr=0.3, lambda_shape=1.5),
    }

    outputs: dict = {}
    fig, axes = plt.subplots(1, len(PRESETS) + 2, figsize=(4 * (len(PRESETS) + 2), 4))
    axes[0].imshow(src_img); axes[0].set_title("Source\n(Identity)"); axes[0].axis("off")
    axes[1].imshow(tgt_img); axes[1].set_title("Target\n(Pose/Expr)"); axes[1].axis("off")

    for ax, (name, params) in zip(axes[2:], PRESETS.items()):
        out = run_pipeline(src_img, tgt_img, verbose=False, **params)
        outputs[name] = out
        id_s = f"{out['identity_sim']:.3f}" if out["identity_sim"] else "N/A"
        ax.imshow(out["result"])
        ax.set_title(f"{name}\nID-sim={id_s}", fontsize=9)
        ax.axis("off")

    plt.suptitle("Explicit Attribute Control — LDFaceNet-Lite + BFM", fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Preset comparison saved → {save_path}")
    plt.show()
    return outputs


# ── Multi-pair ablation ────────────────────────────────────────────────────────

def run_multi_pair_ablation(
    pairs: list,
    configs: Optional[dict] = None,
) -> dict:
    """Run ablation over multiple (src, tgt) pairs and average metrics."""
    if configs is None:
        configs = ABLATION_CONFIGS

    arcface  = _models["arcface"]
    averaged = {title: {"id_sim": [], "ssim": [], "l2": []} for title in configs}

    for pair_idx, (src, tgt) in enumerate(pairs):
        print(f"\n── Pair {pair_idx + 1}/{len(pairs)} ──")
        for title, config in configs.items():
            out = run_pipeline(
                src, tgt,
                lambda_id    = config["id"],
                lambda_pose  = config["pose"],
                lambda_expr  = config["expr"],
                lambda_shape = config["shape"],
                verbose      = False,
            )
            m = compute_metrics(src, tgt, out["result"], arcface)
            for k in averaged[title]:
                averaged[title][k].append(m[k])

    print(f"\n{'Config':<25} {'ID-sim':>8} {'SSIM':>8}")
    print("─" * 45)
    for title, vals in averaged.items():
        print(
            f"{title:<25} "
            f"{np.nanmean(vals['id_sim']):>8.4f} "
            f"{np.nanmean(vals['ssim']):>8.4f}"
        )
    return averaged


# ── Interactive Jupyter demo ───────────────────────────────────────────────────

def interactive_demo(src_img: np.ndarray, tgt_img: np.ndarray):
    """Launch an ipywidgets slider UI. Run inside a Jupyter notebook cell."""
    try:
        import ipywidgets as widgets
        from IPython.display import display
    except ImportError:
        raise ImportError("pip install ipywidgets")

    slider_id    = widgets.FloatSlider(value=2.5, min=0.0, max=5.0, step=0.1,
                                       description="Identity:", continuous_update=False)
    slider_pose  = widgets.FloatSlider(value=0.3, min=0.0, max=2.0, step=0.05,
                                       description="Pose:", continuous_update=False)
    slider_expr  = widgets.FloatSlider(value=0.3, min=0.0, max=2.0, step=0.05,
                                       description="Expression:", continuous_update=False)
    slider_shape = widgets.FloatSlider(value=1.2, min=0.0, max=3.0, step=0.1,
                                       description="Shape:", continuous_update=False)
    slider_str   = widgets.FloatSlider(value=0.55, min=0.3, max=0.9, step=0.05,
                                       description="Strength:", continuous_update=False)

    btn_pose = widgets.Button(description="Pose Only",       button_style="info")
    btn_expr = widgets.Button(description="Expression Only", button_style="warning")
    btn_id   = widgets.Button(description="Fix Identity",    button_style="danger")

    def set_pose_only(_):
        slider_id.value=2.5; slider_pose.value=1.5
        slider_expr.value=0.0; slider_shape.value=0.0

    def set_expr_only(_):
        slider_id.value=2.5; slider_pose.value=0.0
        slider_expr.value=1.5; slider_shape.value=0.0

    def set_fix_id(_):
        slider_id.value=4.0; slider_pose.value=0.3
        slider_expr.value=0.3; slider_shape.value=1.5

    btn_pose.on_click(set_pose_only)
    btn_expr.on_click(set_expr_only)
    btn_id.on_click(set_fix_id)

    run_btn    = widgets.Button(description="▶ Run Swap", button_style="success")
    output_box = widgets.Output()

    ui = widgets.VBox([
        widgets.HTML("<h3>🎭 LDFaceNet Interactive Demo</h3>"),
        widgets.HTML("<b>Presets:</b>"),
        widgets.HBox([btn_pose, btn_expr, btn_id]),
        widgets.HTML("<b>Manual controls:</b>"),
        slider_id, slider_pose, slider_expr, slider_shape, slider_str,
        run_btn, output_box,
    ])

    def on_run(_):
        output_box.clear_output()
        with output_box:
            print("⏳ Running pipeline …")
            try:
                out = run_pipeline(
                    source_img    = src_img,
                    target_img    = tgt_img,
                    lambda_id     = slider_id.value,
                    lambda_pose   = slider_pose.value,
                    lambda_expr   = slider_expr.value,
                    lambda_shape  = slider_shape.value,
                    strength      = slider_str.value,
                    num_steps     = 30,
                    verbose       = False,
                )
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                axes[0].imshow(src_img); axes[0].set_title("Source"); axes[0].axis("off")
                axes[1].imshow(tgt_img); axes[1].set_title("Target"); axes[1].axis("off")
                id_s = f"{out['identity_sim']:.3f}" if out["identity_sim"] else "N/A"
                axes[2].imshow(out["result"])
                axes[2].set_title(f"Result  ID-sim={id_s}", color="green")
                axes[2].axis("off")
                plt.tight_layout()
                plt.show()
            except Exception as exc:
                print(f"❌ {exc}")

    run_btn.on_click(on_run)
    display(ui)
