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
