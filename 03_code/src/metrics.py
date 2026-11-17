from skimage.metrics import structural_similarity as ssim
import cv2
import numpy as np
from typing import Optional, Dict

from src.extractors.arcface import ArcFaceIdentityExtractor

def compute_metrics(src_img: np.ndarray, tgt_img: np.ndarray, result_img: np.ndarray, arcface: Optional[ArcFaceIdentityExtractor] = None) -> Dict[str, float]:
    """Compute ID-similarity (ArcFace), SSIM, and L2 distance for face swap result."""
    if arcface is None:
        try:
            from src.pipeline import _models
            if "arcface" in _models:
                arcface = _models["arcface"]
            else:
                arcface = ArcFaceIdentityExtractor()
        except Exception:
            arcface = ArcFaceIdentityExtractor()

    # 1. ID-sim
    try:
        src_id  = arcface.extract(src_img)
        res_id  = arcface.extract(result_img)
        id_sim  = arcface.cosine_sim(src_id, res_id)
    except Exception:
        id_sim  = float('nan')

    # 2. SSIM — structural similarity to target
    tgt_resized = cv2.resize(tgt_img,    (256, 256))
    res_resized = cv2.resize(result_img, (256, 256))
    ssim_score  = float(ssim(tgt_resized, res_resized, channel_axis=2, data_range=255))

    # 3. Pixel space L2 distance
    l2 = float(np.mean((tgt_resized.astype(float) - res_resized.astype(float)) ** 2))

    return {'id_sim': id_sim, 'ssim': ssim_score, 'l2': l2}
