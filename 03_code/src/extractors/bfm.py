import types
import numpy as np
import torch
import cv2
import face_alignment
from scipy.spatial.transform import Rotation

from ..config import cfg

FACE_3D_MODEL = np.array([
    [ 0.000,   0.000,   0.000],   # nose tip        lmk 30
    [ 0.000, -330.000, -65.000],   # chin             lmk 8
    [-225.000, 170.000,-135.000],  # left eye corner  lmk 36
    [ 225.000, 170.000,-135.000],  # right eye corner lmk 45
    [-150.000,-150.000,-125.000],  # left mouth       lmk 48
    [ 150.000,-150.000,-125.000]   # right mouth      lmk 54
], dtype=np.float64)

class BFMExtractor:
    """
    Landmark-based BFM coefficient approximation using scipy only.
    No face3d / BFM.mat required.
    """
    N_SHAPE = cfg.bfm_n_shape
    N_EXPR  = cfg.bfm_n_expr

    def __init__(self, device=None):
        self.device = device or cfg.device
        self.fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D,
            flip_input=False, device='cpu')

    def _fit_bfm(self, lmks_2d, img_hw):
        H, W = img_hw
        x = lmks_2d.copy().astype(np.float32)
        x[:, 0] = (x[:, 0] / W) * 2 - 1
        x[:, 1] = (x[:, 1] / H) * 2 - 1

        flat = x.flatten()
        sp = np.zeros(self.N_SHAPE, dtype=np.float32)
        sp[:min(self.N_SHAPE, len(flat))] = flat[:self.N_SHAPE]

        diffs = (lmks_2d - lmks_2d.mean(axis=0)).flatten()
        ep = np.zeros(self.N_EXPR, dtype=np.float32)
        ep[:min(self.N_EXPR, len(diffs))] = diffs[:self.N_EXPR] / (max(H, W) + 1e-6)

        return sp, ep

    def _estimate_pose(self, lmks_2d, img_hw):
        """Robust solvePnP using standard 6-point model."""
        H, W = img_hw
        image_pts = np.array([
            lmks_2d[30],   # nose tip
            lmks_2d[8],    # chin
            lmks_2d[36],   # left eye
            lmks_2d[45],   # right eye
            lmks_2d[48],   # left mouth
            lmks_2d[54],   # right mouth
        ], dtype=np.float64)

        focal   = float(W)
        cam_mat = np.array([
            [focal, 0,     W / 2],
            [0,     focal, H / 2],
            [0,     0,     1    ]
        ], dtype=np.float64)

        ok, rvec, tvec = cv2.solvePnP(
            FACE_3D_MODEL, image_pts, cam_mat,
            np.zeros((4, 1)),
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not ok:
            return np.zeros(6, dtype=np.float32)

        rmat, _ = cv2.Rodrigues(rvec)
        euler   = Rotation.from_matrix(rmat).as_euler('yxz', degrees=True)
        yaw, pitch, roll = euler

        if abs(yaw) > 90:
            yaw = yaw - np.sign(yaw) * 180

        return np.array([yaw, pitch, roll,
                         float(tvec[0]), float(tvec[1]), float(tvec[2])],
                        dtype=np.float32)

    def _estimate_illumination(self, image_rgb, lmks_2d):
        H, W = image_rgb.shape[:2]
        cx   = int(lmks_2d[:, 0].mean())
        cy   = int(lmks_2d[:, 1].mean())
        r    = int(min(H, W) * 0.2)
        patch = image_rgb[max(0,cy-r):min(H,cy+r),
                          max(0,cx-r):min(W,cx+r)].astype(np.float32) / 255.0
        fp    = patch.reshape(-1, 3)
        if len(fp) == 0:
            return np.zeros(27, dtype=np.float32)
        sh      = np.zeros(27, dtype=np.float32)
        sh[0:3] = fp.mean(axis=0)
        sh[3:6] = fp.std(axis=0)
        gray    = fp.mean(axis=1)
        sh[6]   = float(np.percentile(gray, 90) - np.percentile(gray, 10))
        sh[7]   = float(np.percentile(gray, 75))
        sh[8]   = float(np.percentile(gray, 25))
        return sh

    @torch.no_grad()
    def extract(self, image_rgb: np.ndarray) -> dict:
        lmks_list = self.fa.get_landmarks(image_rgb)
        if not lmks_list:
            raise ValueError('No face landmarks detected!')
        lmks_2d = lmks_list[0]

        sp, ep = self._fit_bfm(lmks_2d, image_rgb.shape[:2])

        return {
            'shape_coeff' : torch.tensor(sp, dtype=torch.float32).to(self.device),
            'expr_coeff'  : torch.tensor(ep, dtype=torch.float32).to(self.device),
            'pose'        : torch.tensor(
                self._estimate_pose(lmks_2d, image_rgb.shape[:2]),
                dtype=torch.float32).to(self.device),
            'expression'  : torch.tensor(ep, dtype=torch.float32).to(self.device),
            'illumination': torch.tensor(
                self._estimate_illumination(image_rgb, lmks_2d),
                dtype=torch.float32).to(self.device),
            'landmarks'   : lmks_2d,
            'landmarks_3d': None,
        }
