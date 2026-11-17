import numpy as np
import torch
import torch.nn.functional as F
import cv2
import insightface
from insightface.app import FaceAnalysis

from ..config import cfg

class ArcFaceIdentityExtractor:
    """
    Extracts a 512-d L2-normalised identity embedding.
    Uses buffalo_sc (small/fast).
    """
    def __init__(self, device=None, model_name=None, det_size=None):
        self.device = device or cfg.device
        model_name = model_name or cfg.arcface_model_name
        det_size = det_size or cfg.arcface_det_size

        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device == 'cuda' else ['CPUExecutionProvider']
        self.app = FaceAnalysis(name=model_name, providers=providers)
        self.app.prepare(ctx_id=0 if self.device == 'cuda' else -1, det_size=det_size)

    @torch.no_grad()
    def extract(self, image: np.ndarray) -> torch.Tensor:
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        faces   = self.app.get(img_bgr)
        if len(faces) == 0:
            raise ValueError('No face detected!')
        face = sorted(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]),
                      reverse=True)[0]
        emb = torch.tensor(face.normed_embedding, dtype=torch.float32)
        return emb.unsqueeze(0).to(self.device)  # (1, 512)

    def cosine_sim(self, a: torch.Tensor, b: torch.Tensor) -> float:
        return F.cosine_similarity(a, b).item()
