import numpy as np
import torch
import torch.nn.functional as F
import cv2
import torchvision.transforms as T
from PIL import Image

from ..config import cfg

def get_img_transform(img_size=None):
    img_size = img_size or cfg.img_size
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize([0.5]*3, [0.5]*3),
    ])

def encode_image(image_rgb, vae):
    """
    Encode an RGB image (numpy array or PIL Image) into VAE latent space.
    """
    if isinstance(image_rgb, np.ndarray):
        pil = Image.fromarray(image_rgb)
    else:
        pil = image_rgb

    vae_dtype = next(vae.parameters()).dtype
    device = next(vae.parameters()).device

    tf = get_img_transform(cfg.img_size)
    x = tf(pil).unsqueeze(0).to(device, dtype=vae_dtype)

    with torch.no_grad():
        z = vae.encode(x).latent_dist.mean

    return z * vae.config.scaling_factor

def decode_latent(z: torch.Tensor, vae) -> np.ndarray:
    """
    Decode VAE latent tensor back to HxWx3 uint8 numpy RGB image.
    """
    vae_dtype = next(vae.parameters()).dtype
    z_dec = z.to(torch.float32) / vae.config.scaling_factor
    with torch.no_grad():
        vae_f32 = vae.to(torch.float32)
        x = vae_f32.decode(z_dec).sample
        vae.to(vae_dtype)
    x = (x.clamp(-1,1) + 1) / 2
    return (x[0].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)

def mask_to_latent_mask(face_mask: np.ndarray, device=None) -> torch.Tensor:
    device = device or cfg.device
    target_latent_size = face_mask.shape[0] // 8
    m = cv2.resize(face_mask.astype(np.float32),
                   (target_latent_size, target_latent_size), interpolation=cv2.INTER_AREA)
    m = cv2.GaussianBlur(m, (3, 3), 1.0)
    m = np.clip(m, 0, 1)
    return torch.tensor(m).unsqueeze(0).unsqueeze(0).to(device)

def latent_blend_and_decode(z_swap: torch.Tensor, z_target: torch.Tensor, face_mask, vae, blend_hardness: float = 0.5) -> torch.Tensor:
    """
    Blend swapped latent with target latent using face mask, then decode.
    """
    LATENT = z_swap.shape[-1]
    device = z_swap.device

    if isinstance(face_mask, torch.Tensor):
        fm = face_mask.squeeze().cpu().float().numpy()
    elif isinstance(face_mask, np.ndarray):
        fm = face_mask.squeeze().astype(np.float32)
        if fm.max() > 1.0:
            fm = fm / 255.0
    else:
        fm = np.array(face_mask.convert("L")).astype(np.float32) / 255.0

    fm_tensor = torch.tensor(fm, dtype=torch.float32)
    while fm_tensor.ndim < 4:
        fm_tensor = fm_tensor.unsqueeze(0)

    M = F.interpolate(fm_tensor, size=(LATENT, LATENT),
                      mode='bilinear', align_corners=False)
    M = M.to(device)

    M = torch.clamp((M - 0.5) * (1 + blend_hardness) + 0.5, 0, 1)

    z_blend = M * z_swap.float() + (1 - M) * z_target.float()

    vae_dtype = next(vae.parameters()).dtype
    with torch.no_grad():
        decoded = vae.to(torch.float32).decode(z_blend / vae.config.scaling_factor).sample
        vae.to(vae_dtype)

    return decoded
