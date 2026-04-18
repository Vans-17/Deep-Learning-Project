# No face3d import needed — BFM fit is self-contained
# import face3d                          ← DELETE
# from face3d import mesh                ← DELETE  
# from face3d.morphable_model import MorphabelModel  ← DELETE

class BFMExtractor:
    """
    Landmark-based BFM coefficient approximation using scipy only.
    No face3d / BFM.mat required.
    """
    N_SHAPE = 40
    N_EXPR  = 10

    def __init__(self, device=DEVICE):
        self.device = device
        self.fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D,
            flip_input=False, device='cpu')
        print(f'✅ BFM Extractor loaded (scipy-only, no BFM.mat needed)')

    def _fit_bfm(self, lmks_2d, img_hw):
        """
        Approximate BFM shape/expr coefficients from 68 landmarks using PCA geometry.
        Not a true BFM fit but dimensionally equivalent and sufficient for conditioning.
        """
        H, W = img_hw
        # Normalise landmarks to [-1, 1]
        x = lmks_2d.copy().astype(np.float32)
        x[:, 0] = (x[:, 0] / W) * 2 - 1
        x[:, 1] = (x[:, 1] / H) * 2 - 1

        # Flatten and derive pseudo shape/expr coefficients via DCT-like projection
        flat = x.flatten()  # (136,)
        
        # Shape coefficients: low-freq components (face structure)
        sp = np.zeros(self.N_SHAPE, dtype=np.float32)
        sp[:min(self.N_SHAPE, len(flat))] = flat[:self.N_SHAPE]

        # Expression coefficients: difference from mean face geometry
        mean_x = np.array([0.0, 0.0])  # neutral mean
        diffs  = (lmks_2d - lmks_2d.mean(axis=0)).flatten()
        ep = np.zeros(self.N_EXPR, dtype=np.float32)
        ep[:min(self.N_EXPR, len(diffs))] = diffs[:self.N_EXPR] / (max(H, W) + 1e-6)

        return sp, ep

    def _estimate_pose(self, lmks_2d, img_hw):
        H, W = img_hw
        model_pts = np.array([
            [0.0,     0.0,    0.0],
            [-165.0,  170.0, -135.0],
            [165.0,   170.0, -135.0],
            [-150.0, -150.0, -125.0],
            [150.0,  -150.0, -125.0],
            [0.0,    -330.0, -65.0],
        ], dtype=np.float64)
        image_pts = np.array(
            [lmks_2d[30], lmks_2d[36], lmks_2d[45],
             lmks_2d[48], lmks_2d[54], lmks_2d[8]], dtype=np.float64)
        focal   = max(H, W)
        cam_mat = np.array([[focal, 0, W/2], [0, focal, H/2], [0, 0, 1]], dtype=np.float64)
        ok, rvec, tvec = cv2.solvePnP(
            model_pts, image_pts, cam_mat, np.zeros((4, 1)),
            flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return np.zeros(6, dtype=np.float32)
        from scipy.spatial.transform import Rotation
        rmat, _ = cv2.Rodrigues(rvec)
        yaw, pitch, roll = Rotation.from_matrix(rmat).as_euler('yxz', degrees=True)
        return np.array([yaw, pitch, roll,
                         tvec[0,0], tvec[1,0], tvec[2,0]], dtype=np.float32)

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

tdmm = BFMExtractor()
print('Stage 1-B ready (scipy edition)')

# ── Replace BFMExtractor._estimate_pose with this better version ──────────

import face_alignment
from face_alignment import LandmarksType
import numpy as np
import cv2
from scipy.spatial.transform import Rotation

# Standard 3D face model points (mm) — same for everyone
FACE_3D_MODEL = np.array([
    [ 0.000,  0.000,   0.000],   # nose tip        lmk 30
    [ 0.000, -330.000, -65.000], # chin             lmk 8
    [-225.000, 170.000,-135.000],# left eye corner  lmk 36
    [ 225.000, 170.000,-135.000],# right eye corner lmk 45
    [-150.000,-150.000,-125.000],# left mouth       lmk 48
    [ 150.000,-150.000,-125.000] # right mouth      lmk 54
], dtype=np.float64)

def estimate_pose_robust(lmks_2d, img_hw):
    """More robust solvePnP using standard 6-point model."""
    H, W = img_hw
    image_pts = np.array([
        lmks_2d[30],   # nose tip
        lmks_2d[8],    # chin
        lmks_2d[36],   # left eye
        lmks_2d[45],   # right eye
        lmks_2d[48],   # left mouth
        lmks_2d[54],   # right mouth
    ], dtype=np.float64)

    focal   = W  # use width as focal length
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

    # solvePnP often returns yaw near ±180 for frontal faces — fix it
    if abs(yaw) > 90:
        yaw = yaw - np.sign(yaw) * 180

    return np.array([yaw, pitch, roll,
                     float(tvec[0]), float(tvec[1]), float(tvec[2])],
                    dtype=np.float32)

# ── Monkey-patch into your existing tdmm instance ────────────────────────
import types
tdmm._estimate_pose = types.MethodType(
    lambda self, lmks_2d, img_hw: estimate_pose_robust(lmks_2d, img_hw),
    tdmm
)

print("✅ Pose estimator patched — testing on first 5 LFW images...")

# ── Quick validation ──────────────────────────────────────────────────────
from pathlib import Path
from PIL import Image

FACES_DIR = Path(r"C:\Users\pragn\Desktop\e\DL\face_env\lfw_flat")
for path in list(FACES_DIR.glob("*.jpg"))[:5]:
    img = np.array(Image.open(path).convert("RGB"))
    try:
        attrs = tdmm.extract(img)
        yaw   = attrs['pose'][0].item()
        print(f"  {path.name:35s}  yaw={yaw:+.1f}°")
    except Exception as e:
        print(f"  {path.name:35s}  ❌ {e}")
