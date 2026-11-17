import os
import sys
import subprocess
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from pathlib import Path

from ..config import cfg

class LiteFaceSegmentor:
    """
    Tries to load BiSeNet for accurate segmentation.
    Falls back gracefully to a fast landmark-based ellipse mask.
    """
    FACE_LABELS = list(cfg.bisenet_face_labels)

    def __init__(self, device=None):
        self.device    = device or cfg.device
        self.net       = None
        self.available = False
        self._try_load_bisenet()

    def _try_load_bisenet(self):
        model_path = Path(cfg.bisenet_weights_path)
        if not model_path.exists():
            print('  Trying to download BiSeNet weights...')
            try:
                subprocess.run(
                    ['gdown', '--fuzzy', cfg.bisenet_gdrive_url,
                     '-O', str(model_path)],
                    capture_output=True, timeout=60)
            except Exception:
                pass
        if model_path.exists():
            repo_dir = Path('face-parsing.PyTorch')
            if not repo_dir.exists():
                os.system(f'git clone {cfg.bisenet_repo_url} -q')
            if str(repo_dir) not in sys.path:
                sys.path.insert(0, str(repo_dir))
            try:
                from model import BiSeNet as _BiSeNet
                net = _BiSeNet(n_classes=19)
                net.load_state_dict(torch.load(model_path, map_location='cpu'))
                net.eval()
                self.net       = net
                self.available = True
                print('✅ BiSeNet loaded (CPU idle, GPU during inference)')
                return
            except Exception as e:
                print(f'  BiSeNet load failed ({e}) — using ellipse fallback')
        else:
            print('  BiSeNet unavailable — using fast ellipse mask (fine for coursework)')

    def _ellipse_mask(self, lmks_2d, H, W):
        mask = np.zeros((H, W), dtype=np.uint8)
        cx   = int(lmks_2d[:,0].mean())
        cy   = int(lmks_2d[:,1].mean())
        rx   = int((lmks_2d[:,0].max() - lmks_2d[:,0].min()) * 0.6)
        ry   = int((lmks_2d[:,1].max() - lmks_2d[:,1].min()) * 0.65)
        cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 1, -1)
        return mask.astype(bool)

    @torch.no_grad()
    def parse(self, image_rgb: np.ndarray, lmks_2d=None):
        H, W = image_rgb.shape[:2]
        if not self.available:
            fm = self._ellipse_mask(lmks_2d, H, W) if lmks_2d is not None \
                 else np.zeros((H, W), bool)
            return np.zeros((H,W), np.int32), fm, np.zeros((H,W), bool)

        net = self.net.to(self.device)
        tf  = T.Compose([T.ToPILImage(), T.Resize((512,512)), T.ToTensor(),
                         T.Normalize([.485,.456,.406],[.229,.224,.225])])
        x   = tf(image_rgb).unsqueeze(0).to(self.device)
        out = net(x)[0]
        seg = out.argmax(1).squeeze().cpu().numpy().astype(np.int32)
        seg = cv2.resize(seg.astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST)
        net.to('cpu')
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        face_mask = np.isin(seg, self.FACE_LABELS)
        hair_mask = (seg == 17)
        return seg.astype(np.int32), face_mask, hair_mask
