@torch.no_grad()
from src.extractors.arcface import ArcFaceIdentityExtractor
from src.extractors.bfm import BFMExtractor
from src.extractors.clip import CLIPDisentangler
from src.extractors.segmentor import LiteFaceSegmentor
from src.adaptors.identity_projector import IdentityProjector
from src.adaptors.illumination_adain import IlluminationAdain
from src.adaptors.pose_expr_adaptors import PoseExpressionAdapter
from src.diffusion.denoiser import GuidedDenoiser 

def run_pipeline_v2(
    source_img, target_img,
    lambda_id=2.5,
    lambda_pose=0.3,
    lambda_shape=1.2,
    lambda_expr=0.3,
    lambda_light=0.5,
    strength=0.65,
    num_steps=50,
    blend_hardness=0.5, verbose=True
):
    if verbose: print('━━ Stage 1: Attribute Extraction (BFM) ━━')

    src_id_emb = arcface.extract(source_img)
    src_bfm    = tdmm.extract(source_img)
    tgt_bfm    = tdmm.extract(target_img)

    # ── Pose validation (warn only — no longer blocks) ───────────────────
    def check_pose_valid(tdmm_result, label='face', yaw_thresh=90):
        yaw = tdmm_result['pose'][0].item()
        if abs(yaw) > yaw_thresh:
            print(f"  ⚠️  {label} yaw={yaw:.1f}° is extreme (>{yaw_thresh}°) — safe_pose will clamp it")

    check_pose_valid(src_bfm, label='Source')
    check_pose_valid(tgt_bfm, label='Target')

    # ── Pose clamp ───────────────────────────────────────────────────────
    def safe_pose(pose_tensor, max_yaw=45, max_pitch=30):
        p = pose_tensor.clone()
        p[0] = p[0].clamp(-max_yaw,   max_yaw)
        p[1] = p[1].clamp(-max_pitch, max_pitch)
        return p

    lmks_2d = tgt_bfm['landmarks']
    _, tgt_face_mask, _ = bisenet.parse(target_img, lmks_2d)

    src_shape = src_bfm['shape_coeff'].unsqueeze(0)          # (1, 40)
    tgt_pose  = safe_pose(tgt_bfm['pose']).unsqueeze(0)      # (1, 6)  ← clamped
    tgt_expr  = tgt_bfm['expr_coeff'].unsqueeze(0)           # (1, 10)
    tgt_sh    = tgt_bfm['illumination'].unsqueeze(0)         # (1, 27)

    if verbose:
        print(f'  src shape norm : {src_shape.norm():.3f}')
        print(f'  tgt pose yaw   : {tgt_bfm["pose"][0].item():.1f}°  →  clamped to {tgt_pose[0,0].item():.1f}°')
        print(f'  tgt expr norm  : {tgt_expr.norm():.3f}')
        print('  Stage 1 ✓')

    if verbose: print('━━ Stage 2: CLIP Disentangle + Project ━━')
    src_clip_feats = clip_disentangler.extract(source_img)
    clip_id_feat   = src_clip_feats['id_feat'].to(DEVICE)
    id_tokens      = id_projector(src_id_emb, clip_id_feat, lambda_id=lambda_id)

    pose_expr_out = pose_expr_adapter(
        pose_6d   = tgt_pose.to(DEVICE),
        shape_40d = src_shape.to(DEVICE),
        expr_10d  = tgt_expr.to(DEVICE),
        lambda_pose  = lambda_pose,
        lambda_shape = lambda_shape,
        lambda_expr  = lambda_expr,
    )
    if verbose: print('  Stage 2 ✓')

    if verbose: print('━━ Stage 3: VAE Encode ━━')
    z_target = encode_image(target_img)
    if verbose: print(f'  z_target: min={z_target.min():.3f} max={z_target.max():.3f}')

    if verbose: print('━━ Stage 4: img2img Denoise ━━')
    den_sch = DDIMScheduler.from_config(scheduler.config)
    den_sch.set_timesteps(num_steps)

    start_step = int(num_steps * (1 - strength))
    timesteps  = den_sch.timesteps[start_step:]
    t_start    = timesteps[0]

    noise = torch.randn_like(z_target)
    a_t   = den_sch.alphas_cumprod[t_start].to(DEVICE)
    z     = a_t.sqrt() * z_target.float() + (1 - a_t).sqrt() * noise

    if verbose: print(f'  Starting at t={t_start.item()}, {len(timesteps)} steps')

    residuals = pose_expr_out['residuals']
    for step_idx, t in enumerate(timesteps):
        if verbose and step_idx % 5 == 0:
            print(f'  Step {step_idx+1}/{len(timesteps)}')

        cond_hs   = torch.cat([NULL_EMB.expand(z.shape[0], -1, -1).to(DTYPE),
                               id_tokens.to(DTYPE)], dim=1)
        uncond_hs = torch.zeros_like(cond_hs)
        cfg_hs    = torch.cat([uncond_hs, cond_hs])

        latent_input   = torch.cat([z.half()] * 2)
        noise_pred_all = unet(latent_input, t, encoder_hidden_states=cfg_hs).sample
        noise_uncond, noise_cond = noise_pred_all.chunk(2)
        noise_pred = noise_uncond + 3.5 * (noise_cond - noise_uncond)
        noise_pred = noise_pred.float()

        if step_idx < (len(timesteps) * 0.9):
            if residuals is not None:
                res = residuals[0].to(torch.float32).mean(dim=1, keepdim=True).expand_as(noise_pred)
                noise_pred = noise_pred + (pose_expr_adapter.INJECTION_SCALE * lambda_pose) * \
                             torch.clamp(res, -0.5, 0.5)

        z = den_sch.step(noise_pred, t, z).prev_sample.float()
        z = z.clamp(-8, 8)

    if verbose: print(f'  z_denoised: min={z.min():.3f} max={z.max():.3f}')

    if verbose: print('━━ Stage 5: Blend + Decode ━━')
    result_tensor = latent_blend_and_decode(z, z_target, tgt_face_mask, blend_hardness)

    result = (result_tensor.float().clamp(-1, 1) + 1) / 2
    result = (result[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

    try:
        res_id = arcface.extract(result)
        id_sim = arcface.cosine_sim(src_id_emb, res_id)
        if verbose: print(f'  ID-sim (ArcFace cosine): {id_sim:.4f}')
    except Exception:
        id_sim = None

    globals().update({'z_target': z_target, 'z_denoised': z,
                      'tgt_face_mask': tgt_face_mask,
                      'src_bfm_attrs': src_bfm, 'tgt_bfm_attrs': tgt_bfm})

    if verbose: print('✅ Done')
    return {
        'result'      : result,
        'identity_sim': id_sim,
        'attributes'  : tgt_bfm,
        'src_shape'   : src_shape.cpu(),
        'tgt_expr'    : tgt_expr.cpu(),
    }




