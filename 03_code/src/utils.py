# src/utils.py
import csv
import os
import random
from pathlib import Path
from typing import List, Tuple, Optional, Union, Dict, Any
import numpy as np
from PIL import Image

def load_image(path: Union[str, Path]) -> np.ndarray:
    """Load image from path and return as HxWx3 uint8 RGB numpy array."""
    img = Image.open(path).convert("RGB")
    return np.array(img)

def write_csv(rows: List[Dict[str, Any]], csv_path: Union[str, Path]) -> None:
    """Write list of dictionaries to a CSV file."""
    if not rows:
        return
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def pick_frontal(
    pool: List[Path],
    exclude_paths: Optional[List[Path]] = None,
    max_yaw: float = 40.0,
    tdmm: Any = None,
) -> Tuple[Optional[np.ndarray], Optional[Path], Optional[float]]:
    """Find a frontal face image from a pool of paths using BFM pose estimation."""
    if exclude_paths is None:
        exclude_paths = []

    if tdmm is None:
        try:
            from src.extractors.bfm import BFMExtractor
            tdmm = BFMExtractor()
        except Exception as e:
            tdmm = None

    for path in pool:
        if path in exclude_paths:
            continue
        try:
            img = load_image(path)
            if tdmm is not None:
                attrs = tdmm.extract(img)
                yaw = float(attrs['pose'][0].item())
                if abs(yaw) <= max_yaw:
                    return img, path, yaw
            else:
                return img, path, 0.0
        except Exception:
            continue
    return None, None, None

def pick_n_frontal_pairs(
    faces_dir: Union[str, Path],
    n_pairs: int = 3,
    max_yaw: float = 40.0,
    max_attempts: int = 500,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Pick n pairs of frontal face images from faces_dir."""
    faces_dir = Path(faces_dir)
    all_imgs = (
        list(faces_dir.glob("*.jpg"))
        + list(faces_dir.glob("*.jpeg"))
        + list(faces_dir.glob("*.png"))
    )
    if not all_imgs:
        raise RuntimeError(f"No images found in {faces_dir}")

    random.shuffle(all_imgs)
    pool = all_imgs[:max_attempts]

    tdmm = None
    try:
        from src.extractors.bfm import BFMExtractor
        tdmm = BFMExtractor()
    except Exception:
        pass

    used: List[Path] = []
    pairs: List[Tuple[np.ndarray, np.ndarray]] = []
    for _ in range(n_pairs):
        src_img, src_path, _ = pick_frontal(pool, exclude_paths=used, max_yaw=max_yaw, tdmm=tdmm)
        if src_path:
            used.append(src_path)
        tgt_img, tgt_path, _ = pick_frontal(pool, exclude_paths=used, max_yaw=max_yaw, tdmm=tdmm)
        if tgt_path:
            used.append(tgt_path)

        if src_img is None or tgt_img is None:
            break
        pairs.append((src_img, tgt_img))

    return pairs
