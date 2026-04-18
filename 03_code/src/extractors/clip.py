 
import numpy as np
import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPVisionModelWithProjection
from ..config import cfg
 
class CLIPDisentangler(nn.Module):
    """
    Same as original but:
    - Backbone loaded in fp16
    - Moved to CPU between calls to free VRAM
    - Projection dims halved (POSE/EXPR: 64→32, LIGHT: 32→16, ID: 256→128)
    """
    CLIP_DIM   = 512
    POSE_DIM   = 32
    EXPR_DIM   = 32
    LIGHT_DIM  = 16
    ID_DIM     = 128

    def __init__(self, device=DEVICE):
        super().__init__()
        self.device = device
        # Load backbone on CPU in fp16 to save VRAM
        self.clip = CLIPVisionModelWithProjection.from_pretrained(
            'openai/clip-vit-base-patch32',
            torch_dtype=torch.float16
        ).eval()
        for p in self.clip.parameters():
            p.requires_grad_(False)
        self.processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')

        # Smaller projections
        total = self.POSE_DIM + self.EXPR_DIM + self.LIGHT_DIM + self.ID_DIM
        assert total <= self.CLIP_DIM
        self.proj_pose  = nn.Linear(self.CLIP_DIM, self.POSE_DIM,  bias=False)
        self.proj_expr  = nn.Linear(self.CLIP_DIM, self.EXPR_DIM,  bias=False)
        self.proj_light = nn.Linear(self.CLIP_DIM, self.LIGHT_DIM, bias=False)
        self.proj_id    = nn.Linear(self.CLIP_DIM, self.ID_DIM,    bias=False)
        self._init_orthogonal_weights()
        # Keep projections in fp32 on device; CLIP backbone on CPU
        self.to('cpu')
        print('✅ CLIP Disentangler loaded (fp16 backbone, cpu-idle)')

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
        # Temporarily move CLIP to GPU for inference
        self.clip = self.clip.to(self.device)
        pil    = Image.fromarray(image_rgb)
        inputs = self.processor(images=pil, return_tensors='pt')
        pv     = inputs['pixel_values'].to(self.device, dtype=torch.float16)
        emb    = self.clip(pixel_values=pv).image_embeds.float()  # (1, 512)
        self.clip = self.clip.to('cpu')   # free VRAM
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


clip_disentangler = CLIPDisentangler()
print('Stage 1-D ready')
