import torch
from diffusers import DDIMScheduler

from ..config import cfg

class GuidedDenoiser:
    """
    Guided Denoiser wrapper around UNet for classifier-free guidance
    and scheduled feature injection.
    """
    def __init__(self, unet, NULL_EMB, scheduler_config, injection_scale=0.02, device=None, dtype=None):
        self.unet = unet
        self.NULL_EMB = NULL_EMB
        self.scheduler_config = scheduler_config
        self.injection_scale = injection_scale
        self.device = device or cfg.device
        self.dtype = dtype or cfg.dtype

    @torch.no_grad()
    def denoise(self, z_T, id_tokens, src_id_emb=None, pose_expr_out=None, sh_coeffs=None,
                lambda_id=1.0, lambda_pose=1.0, lambda_expr=1.0, lambda_light=1.0,
                num_steps=50, verbose=True):

        den_sch = DDIMScheduler.from_config(self.scheduler_config)
        den_sch.set_timesteps(num_steps)

        z = z_T.clone().to(self.device).to(self.dtype)
        id_tokens = id_tokens.to(self.device).to(self.dtype)
        cond_hs = torch.cat([self.NULL_EMB.expand(z.shape[0], -1, -1).to(self.dtype), id_tokens], dim=1)

        residuals = pose_expr_out['residuals'] if (pose_expr_out and 'residuals' in pose_expr_out) else None

        guidance_scale = cfg.cfg_scale

        for step_idx, t in enumerate(den_sch.timesteps):
            latent_model_input = torch.cat([z] * 2)

            uncond_hs = torch.zeros_like(cond_hs)
            cfg_hs = torch.cat([uncond_hs, cond_hs])

            noise_pred_all = self.unet(latent_model_input, t, encoder_hidden_states=cfg_hs).sample

            noise_pred_uncond, noise_pred_text = noise_pred_all.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            if step_idx < (num_steps * 0.5):
                if residuals is not None:
                    res = residuals[0].to(self.dtype).mean(dim=1, keepdim=True).expand_as(noise_pred)
                    noise_pred = noise_pred + (self.injection_scale * lambda_pose) * torch.clamp(res, -0.15, 0.15)

            z = den_sch.step(noise_pred, t, z).prev_sample
            z = z.clamp(-8, 8)

        return z
