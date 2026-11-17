# ── Cell: Load Source & Target from Google Drive ──────────────────────────
import os, random
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple

from .config import cfg

# ── Mount Google Drive ────────────────────────────────────────────────────
from google.colab import drive
drive.mount('/content/drive')

# ── Point to your shared folder ───────────────────────────────────────────
FACES_DIR    = Path("/content/drive/MyDrive/sample_inputs")  # adjust if folder name differs
MAX_ATTEMPTS = 200
MAX_YAW      = 40

def load_image(path):
    return np.array(Image.open(path).convert("RGB"))

def pick_frontal(all_imgs, exclude_path=None, max_yaw=MAX_YAW):
    for path in all_imgs:
        if exclude_path and path == exclude_path:
            continue
        img = load_image(path)
        try:
            yaw = tdmm.extract(img)['pose'][0].item()
            if abs(yaw) < max_yaw:
                return img, path, yaw
        except:
            continue
    return None, None, None

# ── Collect images ────────────────────────────────────────────────────────
all_imgs = list(FACES_DIR.glob("*.jpg")) + \
           list(FACES_DIR.glob("*.jpeg")) + \
           list(FACES_DIR.glob("*.png"))

if len(all_imgs) == 0:
    raise RuntimeError(f"No images found in {FACES_DIR} — check the folder path after mounting")

print(f"Found {len(all_imgs)} images in Drive folder")
random.shuffle(all_imgs)
pool = all_imgs[:MAX_ATTEMPTS]

# ── Source ────────────────────────────────────────────────────────────────
print("🔍 Sampling frontal source...")
src_img, src_path, src_yaw = pick_frontal(pool)
if src_img is None:
    raise RuntimeError("No frontal source found — try raising MAX_YAW or MAX_ATTEMPTS")
print(f"✅ Source : {src_path.name}  (yaw={src_yaw:.1f}°)")

# ── Target ────────────────────────────────────────────────────────────────
print("🔍 Sampling frontal target...")
tgt_img, tgt_path, tgt_yaw = pick_frontal(pool, exclude_path=src_path)
if tgt_img is None:
    raise RuntimeError("No frontal target found — try raising MAX_YAW or MAX_ATTEMPTS")
print(f"✅ Target : {tgt_path.name}  (yaw={tgt_yaw:.1f}°)")

# ── Preview ───────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].imshow(src_img); axes[0].set_title(f"Source\n{src_path.name}\nyaw={src_yaw:.1f}°"); axes[0].axis('off')
axes[1].imshow(tgt_img); axes[1].set_title(f"Target\n{tgt_path
