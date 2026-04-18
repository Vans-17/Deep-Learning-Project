img_to_tensor = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)), # This must use the variable
    T.ToTensor(),
    T.Normalize([0.5]*3, [0.5]*3),
])

def encode_image(image_rgb):
    # 1. Convert numpy/PIL to PIL
    if isinstance(image_rgb, np.ndarray):
        pil = Image.fromarray(image_rgb)
    else:
        pil = image_rgb
        
    # 2. Get the VAE's current data type (usually float32 or float16)
    # This prevents the "Half vs float" mismatch
    vae_dtype = next(vae.parameters()).dtype
    
    # 3. Convert image to tensor and match the VAE's type
    x = img_to_tensor(pil).unsqueeze(0).to(DEVICE, dtype=vae_dtype)
    
    with torch.no_grad():
        # Encode to latent space
        z = vae.encode(x).latent_dist.mean
        
    return z * vae.config.scaling_factor

def decode_latent(z: torch.Tensor) -> np.ndarray:
    # Force z to float32 before math to avoid NaNs
    z_dec = z.to(torch.float32) / vae.config.scaling_factor
    with torch.no_grad():
        # Temporarily move VAE to float32 for the decode if needed
        x = vae.to(torch.float32).decode(z_dec).sample
    x = (x.clamp(-1,1) + 1) / 2
    # Move VAE back to DTYPE for next run to save VRAM
    vae.to(DTYPE) 
    return (x[0].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)

def mask_to_latent_mask(face_mask: np.ndarray) -> torch.Tensor:
    # Use // 8 to dynamically match the latent resolution (e.g., 192 -> 24)
    target_latent_size = face_mask.shape[0] // 8 
    m = cv2.resize(face_mask.astype(np.float32),
                   (target_latent_size, target_latent_size), interpolation=cv2.INTER_AREA)
    m = cv2.GaussianBlur(m, (3, 3), 1.0)
    m = np.clip(m, 0, 1)
    return torch.tensor(m).unsqueeze(0).unsqueeze(0).to(DEVICE)

print('Stage 3-A helpers ready')
