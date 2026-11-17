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
from typing import Optional, Union, Dict, Any

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
        device=cfg.device,
        dtype=cfg.dtype,
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
    models: Optional[Any]                 = None,
    lambda_id: Optional[float]            = None,
    lambda_pose: Optional[float]          = None,
    lambda_shape: Optional[float]         = None,
    lambda_expr: Optional[float]          = None,
    lambda_light: Optional[float]         = None,
    strength: Optional[float]             = None,
    num_steps: Optional[int]              = None,
    blend_hardness: Optional[float]       = None,
    verbose: bool                         = True,
    init_img: Optional[np.ndarray]        = None,
    **kwargs
) -> dict:
    """
    Run the full LDFaceNet-Lite + BFM face swap pipeline.

    Handles callers passing models dictionary positionally or as keyword argument,
    as well as dictionaries of config parameters.
    """
    global _models

    # Handle parameters if passed as a dictionary in position 3 or 4
    config_dict = {}
    if isinstance(models, dict) and ("arcface" not in models and "vae" not in models):
        config_dict = models
        active_models = _models
    elif isinstance(models, dict):
        active_models = models
    elif isinstance(models, (float, int)):
        if lambda_id is None:
            lambda_id = float(models)
        active_models = _models
    else:
        active_models = _models

    if isinstance(lambda_id, dict):
        config_dict.update(lambda_id)
        lambda_id = None

    if kwargs:
        config_dict.update(kwargs)

    # Resolve default values
    l_id     = float(config_dict.get('lambda_id',     lambda_id    if lambda_id    is not None else cfg.default_lambda_id))
    l_pose   = float(config_dict.get('lambda_pose',   lambda_pose  if lambda_pose  is not None else cfg.default_lambda_pose))
    l_shape  = float(config_dict.get('lambda_shape',  lambda_shape if lambda_shape is not None else cfg.default_lambda_shape))
    l_expr   = float(config_dict.get('lambda_expr',   lambda_expr  if lambda_expr  is not None else cfg.default_lambda_expr))
    l_light  = float(config_dict.get('lambda_light',  lambda_light if lambda_light is not None else cfg.default_lambda_light))
    str_val  = float(config_dict.get('strength',      strength     if strength     is not None else cfg.default_strength))
    steps    = int(config_dict.get('num_steps',       num_steps    if num_steps    is not None else cfg.ddim_steps_denoise))
    b_hard   = float(config_dict.get('blend_hardness', blend_hardness if blend_hardness is not None else cfg.default_blend_hardness))

    if not active_models or "arcface" not in active_models:
        active_models = init_models()

    arcface   = active_models["arcface"]
    tdmm      = active_models["tdmm"]
    bisenet   = active_models["bisenet"]
    clip_dis  = active_models["clip_dis"]
    id_proj   = active_models["id_proj"]
    pose_adap = active_models["pose_adap"]
    vae       = active_models["vae"]
    unet      = active_models["unet"]
    scheduler = active_models["scheduler"]
    NULL_EMB  = active_models["NULL_EMB"]

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
    id_tokens = id_proj(src_id_emb, clip_id, lambda_id=l_id)  # (1, 2, 768)

    pose_expr_out = pose_adap(
        pose_6d   = tgt_pose.to(cfg.device),
        shape_40d = src_shape.to(cfg.device),
        expr_10d  = tgt_expr.to(cfg.device),
        lambda_pose  = l_pose,
        lambda_shape = l_shape,
        lambda_expr  = l_expr,
    )

    if verbose:
        print("  Stage 2 ✓")

    # ── Stage 3: VAE encode target ─────────────────────────────────────────
    if verbose:
        print("━━ Stage 3: VAE Encode ━━")

    # init_img (e.g. warp-swap result) carries the identity; target is the
    # pose/expression donor. Encode whichever seeds the denoiser.
    z_target = encode_image(target_img if init_img is None else init_img, vae)

    if verbose:
        print(f"  z_target: min={z_target.min():.3f} max={z_target.max():.3f}")

    # ── Stage 4: img2img guided denoising ─────────────────────────────────
    if verbose:
        print("━━ Stage 4: img2img Denoise ━━")

    den_sch = DDIMScheduler.from_config(scheduler.config)
    den_sch.set_timesteps(steps)

    start_step = int(steps * (1 - str_val))
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

        cond_hs   = torch.cat(
            [NULL_EMB.expand(z.shape[0], -1, -1).to(cfg.dtype),
             id_tokens.to(cfg.dtype)],
            dim=1,
        )
        uncond_hs = torch.zeros_like(cond_hs)
        cfg_hs    = torch.cat([uncond_hs, cond_hs])

        latent_input   = torch.cat([z.to(cfg.dtype)] * 2)
        noise_pred_all = unet(latent_input, t, encoder_hidden_states=cfg_hs).sample
        noise_uncond, noise_cond = noise_pred_all.chunk(2)
        cfg_scale_eff  = float(config_dict.get('cfg_scale') or cfg.cfg_scale)
        noise_pred = (noise_uncond + cfg_scale_eff * (noise_cond - noise_uncond)).float()

        if cfg.id_guidance_scale > 0:
            idg_hs = torch.cat(
                [NULL_EMB.expand(z.shape[0], -1, -1).to(cfg.dtype),
                 torch.zeros_like(id_tokens.to(cfg.dtype))],
                dim=1,
            )
            noise_noid = unet(
                latent_input[:1], t, encoder_hidden_states=idg_hs
            ).sample.float()
            noise_pred = noise_pred + cfg.id_guidance_scale * (
                noise_cond - noise_noid
            )

        if step_idx < int(len(timesteps) * 0.9) and residuals is not None:
            res = (
                residuals[0]
                .to(torch.float32)
                .mean(dim=1, keepdim=True)
                .expand_as(noise_pred)
            )
            noise_pred = noise_pred + (
                pose_adap.INJECTION_SCALE * l_pose
            ) * torch.clamp(res, -0.5, 0.5)

        z = den_sch.step(noise_pred, t, z, eta=float(config_dict.get('eta', 0.0))).prev_sample.float()
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
        blend_hardness= b_hard,
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


# ── Landmark-based warp swap (classical fallback) ─────────────────────────────

def _skin_statistics_mask(face_mask: np.ndarray, lmks: np.ndarray,
                          H: int, W: int) -> np.ndarray:
    """
    Boolean mask over the inner face region only (cheeks + forehead),
    excluding eyes, eyebrows, mouth and the mask fringe. Used to compute
    color-transfer statistics so hair/background never bias skin tone.
    """
    skin = face_mask.astype(np.uint8).copy()

    # Shrink inward from the ellipse fringe (hair / background edge)
    skin = cv2.erode(skin, np.ones((25, 25), np.uint8))

    # Exclude eyes (36-41, 42-47), brows (17-26) and mouth (48-67)
    for pts in (lmks[17:27], lmks[36:48], lmks[48:68]):
        poly = cv2.convexHull(pts.astype(np.float32).reshape(-1, 1, 2)).astype(np.int32)
        cv2.fillConvexPoly(skin, poly, 0)

    return skin.astype(bool)

def run_warp_swap(
    source_img: np.ndarray,
    target_img: np.ndarray,
    models: Optional[Any] = None,
    verbose: bool = True,
    **kwargs
) -> dict:
    """
    Classical face swap: estimate a similarity transform from the source
    landmarks to the target landmarks, warp the source face into the target
    frame, then Poisson-blend it with the target face mask.

    This reliably transfers identity (unlike the untrained diffusion
    conditioning path, which the UNet largely ignores on CPU).
    """
    global _models
    active_models = models if isinstance(models, dict) and "tdmm" in models else _models
    if not active_models:
        active_models = init_models()

    tdmm    = active_models["tdmm"]
    bisenet = active_models["bisenet"]
    arcface = active_models["arcface"]

    src_lmks = tdmm.extract(source_img)["landmarks"].astype(np.float32)
    tgt_bfm  = tdmm.extract(target_img)
    tgt_lmks = tgt_bfm["landmarks"].astype(np.float32)

    if src_lmks.shape != tgt_lmks.shape:
        raise ValueError("Landmark sets differ — cannot align.")

    H, W = target_img.shape[:2]

    # ── Warp: global similarity (identity-preserving) + optional gentle
    #    piecewise blend for local alignment without distortion ────────────
    method     = kwargs.get('warp_method', 'global')
    warp_scale = float(kwargs.get('warp_scale', 1.0))   # enlarge source face
    pw_weight  = float(kwargs.get('pw_weight', 0.0))    # 0=global only, 1=full piecewise
    warp_dy    = float(kwargs.get('warp_dy', 0.0))      # vertical shift (px, + = up)
    target_skin_weight = float(kwargs.get('target_skin_weight', kwargs.get('skin_tone_weight', 0.0)))
    skin_lightness     = float(kwargs.get('skin_lightness', kwargs.get('lightness', 0.0)))

    # Enlarge the TARGET landmarks outward from their centroid so the fitted
    # similarity transform maps the source face onto a larger region.
    # (Scaling the source landmarks instead has the inverse effect — the
    # fitted transform shrinks the warped face.)
    tgt_fit = tgt_lmks.copy()
    if warp_scale != 1.0:
        c = tgt_fit.mean(axis=0)
        tgt_fit = c + (tgt_fit - c) * warp_scale
    if warp_dy != 0.0:
        tgt_fit[:, 1] -= warp_dy  # shift target landmarks up by warp_dy px

    M, _ = cv2.estimateAffinePartial2D(
        src_lmks, tgt_fit, method=cv2.RANSAC,
        ransacReprojThreshold=5.0, maxIters=2000,
    )
    if M is None:
        raise RuntimeError("Landmark alignment failed.")
    src_global = cv2.warpAffine(
        source_img, M, (W, H), borderMode=cv2.BORDER_REFLECT,
    )

    if pw_weight > 0:
        # Local piecewise alignment, blended in gently so the source's
        # proportions are preserved (reduces piecewise distortion)
        M_glob, inliers = cv2.estimateAffinePartial2D(
            src_lmks, tgt_fit, method=cv2.RANSAC,
            ransacReprojThreshold=5.0, maxIters=2000,
        )
        inlier_mask = inliers.ravel() == 1
        if inlier_mask.sum() < 30:
            inlier_mask = np.ones(len(src_lmks), dtype=bool)

        src_in = src_lmks[inlier_mask]
        tgt_in = tgt_fit[inlier_mask]

        border = np.array([
            [0, 0], [W - 1, 0], [0, H - 1], [W - 1, H - 1],
            [W // 2, 0], [W // 2, H - 1], [0, H // 2], [W - 1, H // 2],
        ], dtype=np.float32)
        src_pts = np.vstack([src_in, border])
        tgt_pts = np.vstack([tgt_in, border])

        from scipy.spatial import Delaunay
        tri = Delaunay(src_pts).simplices

        src_pw = np.zeros_like(target_img)
        for t in tri:
            s = src_pts[t].astype(np.float32)
            d = tgt_pts[t].astype(np.float32)
            if (d[:, 0] < 0).any() or (d[:, 0] >= W).any() or \
               (d[:, 1] < 0).any() or (d[:, 1] >= H).any():
                continue
            A = cv2.getAffineTransform(s, d)
            mask = np.zeros((H, W), np.uint8)
            cv2.fillConvexPoly(mask, d.astype(np.int32), 1)
            warped = cv2.warpAffine(source_img, A, (W, H))
            src_pw[mask == 1] = warped[mask == 1]
        src_pw = src_pw.astype(np.uint8)

        # Blend piecewise result in with weight pw_weight; the global warp
        # remains the base so identity proportions are kept
        src_warped = cv2.addWeighted(src_global, 1.0 - pw_weight,
                                     src_pw, pw_weight, 0)
    else:
        src_warped = src_global

    _, face_mask, _ = bisenet.parse(target_img, tgt_lmks)
    mask_u8 = (face_mask.astype(np.uint8) * 255)

    # ── Color match: fit warped source face lighting to the target ─────────
    # Lab-space Reinhard transfer on SKIN PIXELS ONLY (exclude eyes, brows,
    # mouth and the mask fringe so hair/background don't bias the statistics).
    skin = _skin_statistics_mask(mask_u8, tgt_lmks, H, W)
    m = cv2.GaussianBlur(skin.astype(np.float32), (0, 0), 7)
    m = m / 255.0
    m3 = m[..., None]

    src_lab = cv2.cvtColor(src_warped, cv2.COLOR_RGB2LAB).astype(np.float32)
    tgt_lab = cv2.cvtColor(target_img, cv2.COLOR_RGB2LAB).astype(np.float32)
    matched_lab = src_lab.copy()
    for c in range(3):
        s = src_lab[..., c]
        t = tgt_lab[..., c]
        s_m = (s * m).sum() / m.sum()
        t_m = (t * m).sum() / m.sum()
        s_std = np.sqrt(((s - s_m) ** 2 * m).sum() / m.sum()) + 1e-6
        t_std = np.sqrt(((t - t_m) ** 2 * m).sum() / m.sum()) + 1e-6
        # When target_skin_weight > 0, suppress source color variance and pull skin chrominance toward target
        var_scale = max(0.0, 1.0 - target_skin_weight) if c > 0 else max(0.0, 1.0 - 0.3 * target_skin_weight)
        matched_lab[..., c] = (s - s_m) * (t_std / s_std) * var_scale + t_m
    matched_lab = np.clip(matched_lab, 0, 255).astype(np.uint8)
    matched_rgb = cv2.cvtColor(matched_lab, cv2.COLOR_LAB2RGB).astype(np.float32)

    # Feather the transfer across the face mask so skin tone transitions smoothly
    m_feather = cv2.GaussianBlur(mask_u8.astype(np.float32), (0, 0), 15) / 255.0
    m_feather = m_feather[..., None]
    src_warped = (src_warped.astype(np.float32) * (1 - m_feather)
                  + matched_rgb * m_feather)
    src_warped = np.clip(src_warped, 0, 255).astype(np.uint8)

    # ── Feather mask edges so the blend boundary is invisible ──────────────
    mask_feathered = np.clip(m_feather[..., 0] * 255, 0, 255).astype(np.uint8)

    center = (int(tgt_lmks[:, 0].mean()), int(tgt_lmks[:, 1].mean()))
    result = cv2.seamlessClone(
        src_warped, target_img, mask_feathered, center, cv2.NORMAL_CLONE,
    )

    # ── Cleanup: smooth triangle seams / artifacts inside the face region ──
    # A gentle bilateral filter preserves edges (eyes, mouth) while removing
    # the noise left by the warp / piecewise blending. Applied only inside
    # the feathered mask so the background stays untouched.
    clean = cv2.bilateralFilter(result, 9, 40, 40)
    m3 = m_feather
    result = (result.astype(np.float32) * (1 - m3)
              + clean.astype(np.float32) * m3)
    result = np.clip(result, 0, 255).astype(np.uint8)

    # ── Skin lightness adjustment (make swapped face skin lighter/fairer) ───
    if skin_lightness != 0.0:
        res_lab = cv2.cvtColor(result, cv2.COLOR_RGB2LAB).astype(np.float32)
        l_norm = np.clip(res_lab[..., 0] / 255.0, 1e-4, 1.0)
        gamma = 1.0 / (1.0 + skin_lightness)
        l_light = (l_norm ** gamma) * 255.0
        res_lab[..., 0] = res_lab[..., 0] * (1.0 - m_feather[..., 0]) + l_light * m_feather[..., 0]
        result = cv2.cvtColor(np.clip(res_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)

    try:
        res_id = arcface.extract(result)
        id_sim = arcface.cosine_sim(arcface.extract(source_img), res_id)
    except Exception:
        id_sim = None

    if verbose:
        id_s = f"{id_sim:.4f}" if id_sim is not None else "N/A"
        print(f"  ID-sim (warp swap): {id_s}")

    return {"result": result, "identity_sim": id_sim, "attributes": tgt_bfm,
            "face_mask": m_feather[..., 0]}


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_pipeline(path: Path = cfg.checkpoint_path):
    """Save state dicts of trainable conditioning modules."""
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
    """Restore state dicts of conditioning modules from a checkpoint."""
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

    if not _models:
        init_models()

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
    if n == 1:
        axes = [axes]
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

    if not _models:
        init_models()

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
