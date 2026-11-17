import torch
from diffusers import DDIMScheduler
 
from ..config import cfg

class GuidedDenoiser:
    @torch.no_grad()
    def denoise(self, z_T, id_tokens, src_id_emb, pose_expr_out, sh_coeffs,
                lambda_id=1.0, lambda_pose=1.0, lambda_expr=1.0, lambda_light=1.0,
                num_steps=50, verbose=True):
        
        den_sch = DDIMScheduler.from_config(scheduler.config)
        den_sch.set_timesteps(num_steps)
        
        z = z_T.clone().to(DEVICE).to(DTYPE)
        id_tokens = id_tokens.to(DEVICE).to(DTYPE)
        cond_hs = torch.cat([NULL_EMB.expand(z.shape[0], -1, -1).to(DTYPE), id_tokens], dim=1)
        
        residuals = pose_expr_out['residuals']

        guidance_scale = 5.0 # The "sharpness" factor
        
        for step_idx, t in enumerate(den_sch.timesteps):
            # 1. Double the batch for CFG (Unconditioned + Conditioned)
            latent_model_input = torch.cat([z] * 2)
            
            # 2. Create unconditioned tokens (zeros)
            uncond_hs = torch.zeros_like(cond_hs)
            cfg_hs = torch.cat([uncond_hs, cond_hs])

            # 3. Predict noise for both
            noise_pred_all = unet(latent_model_input, t, encoder_hidden_states=cfg_hs).sample
            
            # 4. Perform Guidance
            noise_pred_uncond, noise_pred_text = noise_pred_all.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # 3. Scheduled Injection (Only first 50% of steps)
            # This allows the last 25 steps to just focus on "Image Quality"
            if step_idx < (num_steps * 0.5):
                if residuals is not None:
                    res = residuals[0].to(DTYPE).mean(dim=1, keepdim=True).expand_as(noise_pred)
                    # Injection weight now 0.02 — safe with normalised BFM coefficients
                    noise_pred = noise_pred + (pose_expr_adapter.INJECTION_SCALE * lambda_pose) * torch.clamp(res, -0.15, 0.15)

            # 4. Step and Clamp
            z = den_sch.step(noise_pred, t, z).prev_sample
            
            # Keep latents in a healthy range to prevent "Gray Spots"
            z = z.clamp(-8, 8) 

        return z


##LATENT BLEND AND DECODE 
import torch.nn.functional as F

def latent_blend_and_decode(z_swap, z_target, face_mask, blend_hardness=0.5):
    LATENT = z_swap.shape[-1]  # 64 for 512px

    # ── Normalise mask to float32 numpy 2D ───────────────────────────────
    if isinstance(face_mask, torch.Tensor):
        fm = face_mask.squeeze().cpu().float().numpy()
    elif isinstance(face_mask, np.ndarray):
        fm = face_mask.squeeze().astype(np.float32)
        if fm.max() > 1.0:
            fm = fm / 255.0
    else:
        # PIL
        fm = np.array(face_mask.convert("L")).astype(np.float32) / 255.0

    # ── Force resize to exact latent size using torch (most reliable) ────
    fm_tensor = torch.tensor(fm, dtype=torch.float32)
    # Ensure 4D for interpolate: (1, 1, H, W)
    while fm_tensor.ndim < 4:
        fm_tensor = fm_tensor.unsqueeze(0)

    M = F.interpolate(fm_tensor, size=(LATENT, LATENT),
                      mode='bilinear', align_corners=False)  # (1, 1, LATENT, LATENT)
    M = M.to(DEVICE)

    # ── Sharpen ───────────────────────────────────────────────────────────
    M = torch.clamp((M - 0.5) * (1 + blend_hardness) + 0.5, 0, 1)

    print(f"  [blend] M:{M.shape} zs:{z_swap.shape} zt:{z_target.shape}")

    # ── Blend ─────────────────────────────────────────────────────────────
    z_blend = M * z_swap.float() + (1 - M) * z_target.float()

    # ── Decode ────────────────────────────────────────────────────────────
    vae.to(torch.float32)
    with torch.no_grad():
        decoded = vae.decode(z_blend / vae.config.scaling_factor).sample

    return decoded

print("✅ latent_blend_and_decode fixed (F.interpolate — handles any mask size)")
