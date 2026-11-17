import torch
from diffusers import DDIMScheduler
 
from ..config import cfg
 
@torch.no_grad()
def ddim_inversion(z_start: torch.Tensor, unet, scheduler, NULL_EMB, num_steps: int = 30) -> torch.Tensor:
    """
    DDIM inversion to reverse-sample latent z_start back to initial noise latent z_T.
    """
    inv_sch = DDIMScheduler.from_config(scheduler.config)
    inv_sch.set_timesteps(num_steps)

    # Use float32 for inversion — fp16 causes numerical drift
    z = z_start.clone().float()

    ts = list(reversed(inv_sch.timesteps))
    for i, t in enumerate(ts):
        noise_pred = unet(
            z.to(cfg.dtype), t,
            encoder_hidden_states=NULL_EMB.expand(z.shape[0], -1, -1)
        ).sample.float()

        a_t = inv_sch.alphas_cumprod[t].to(z.device)
        a_n = inv_sch.alphas_cumprod[ts[i+1]].to(z.device) \
              if i+1 < len(ts) else torch.tensor(1.0, device=z.device)

        pred_x0 = (z - (1 - a_t).sqrt() * noise_pred) / a_t.sqrt()
        pred_x0 = pred_x0.clamp(-4, 4)
        z = a_n.sqrt() * pred_x0 + (1 - a_n).sqrt() * noise_pred
        z = z.clamp(-6, 6)

    return z
