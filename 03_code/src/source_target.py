# src/source_target.py
import os
import random
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Any

from src.utils import load_image, pick_frontal

def sample_source_and_target(
    faces_dir: Path,
    max_attempts: int = 200,
    max_yaw: float = 40.0,
    tdmm: Any = None,
) -> Tuple[np.ndarray, Path, float, np.ndarray, Path, float]:
    """Sample frontal source and target face images from a directory."""
    all_imgs = (
        list(faces_dir.glob("*.jpg"))
        + list(faces_dir.glob("*.jpeg"))
        + list(faces_dir.glob("*.png"))
    )

    if not all_imgs:
        raise RuntimeError(f"No images found in {faces_dir}")

    random.shuffle(all_imgs)
    pool = all_imgs[:max_attempts]

    src_img, src_path, src_yaw = pick_frontal(pool, max_yaw=max_yaw, tdmm=tdmm)
    if src_img is None:
        raise RuntimeError("No frontal source found")

    tgt_img, tgt_path, tgt_yaw = pick_frontal(pool, exclude_paths=[src_path], max_yaw=max_yaw, tdmm=tdmm)
    if tgt_img is None:
        raise RuntimeError("No frontal target found")

    return src_img, src_path, src_yaw, tgt_img, tgt_path, tgt_yaw
