import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import CLIPProcessor, CLIPVisionModelWithProjection
from ..config import cfg

class CLIPDisentangler(nn.Module):
    """
    CLIP Disentangler extracting pose, expr, light, id features.
    """
    CLIP_DIM   = 512
    POSE_DIM   = cfg.clip_pose_dim
    EXPR_DIM   = cfg.clip_expr_dim
    LIGHT_DIM  = cfg.clip_light_dim
    ID_DIM     = cfg.clip_id_dim

    def __init__(self, device=None):
        super().__init__()
        self.device = device or cfg.device
        self.clip = CLIPVisionModelWithProjection.from_pretrained(
            cfg.clip_model_id,
            torch_dtype=cfg.dtype
        ).eval()
        for p in self.clip.parameters():
            p.requires_grad_(False)
        self.processor = CLIPProcessor.from_pretrained(cfg.clip_model_id)

        total = self.POSE_DIM + self.EXPR_DIM + self.LIGHT_DIM + self.ID_DIM
        assert total <= self.CLIP_DIM
        self.proj_pose  = nn.Linear(self.CLIP_DIM, self.POSE_DIM,  bias=False)
        self.proj_expr  = nn.Linear(self.CLIP_DIM, self.EXPR_DIM,  bias=False)
        self.proj_light = nn.Linear(self.CLIP_DIM, self.LIGHT_DIM, bias=False)
        self.proj_id    = nn.Linear(self.CLIP_DIM, self.ID_DIM,    bias=False)
        self._init_orthogonal_weights()
        self.to('cpu')

    def _init_orthogonal_weights(self):
        torch.manual_seed(42)
        basis = torch.linalg.qr(torch.randn(self.CLIP_DIM, self.CLIP_DIM)).Q
        offset = 0
        for proj, dim in [(self.proj_pose,  self.POSE_DIM),
                          (self.proj_expr,  self.EXPR_DIM),
                          (self.proj_light, self.LIGHT_DIM),
                          (self.proj_id,    self.ID_DIM)]:
            proj.weight.data = basis[offset:offset+dim].clone()
            offset += dim

    @torch.no_grad()
    def extract(self, image_rgb: np.ndarray) -> dict:
        self.clip = self.clip.to(self.device)
        pil    = Image.fromarray(image_rgb)
        inputs = self.processor(images=pil, return_tensors='pt')
        pv     = inputs['pixel_values'].to(self.device, dtype=cfg.dtype)
        emb    = self.clip(pixel_values=pv).image_embeds.float()
        self.clip = self.clip.to('cpu')
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        emb = emb.to('cpu')
        return {
            'clip_embedding': emb,
            'pose_feat'     : self.proj_pose(emb),
            'expr_feat'     : self.proj_expr(emb),
            'light_feat'    : self.proj_light(emb),
            'id_feat'       : self.proj_id(emb),
        }

    def orthogonality_loss(self) -> torch.Tensor:
        W    = [self.proj_pose.weight, self.proj_expr.weight,
                self.proj_light.weight, self.proj_id.weight]
        loss = torch.tensor(0.0)
        for i in range(len(W)):
            for j in range(i+1, len(W)):
                loss = loss + (W[i] @ W[j].T).pow(2).mean()
        return loss
