import torch
import torch.nn as nn
import torch.nn.functional as F
 
from ..config import cfg
class IdentityProjector(nn.Module):
    """
    Projects ArcFace + CLIP-id embeddings into cross-attention token space.
    Lite: num_tokens 4→2, hidden 1024→512
    """
    def __init__(self, arcface_dim=512, clip_id_dim=128,
                 cross_attn_dim=768, num_tokens=2, device=DEVICE):
        super().__init__()
        self.num_tokens     = num_tokens
        self.cross_attn_dim = cross_attn_dim

        self.arc_proj = nn.Sequential(
            nn.Linear(arcface_dim, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Linear(512, num_tokens * cross_attn_dim),
        )
        self.clip_proj = nn.Sequential(
            nn.Linear(clip_id_dim, 256),
            nn.GELU(),
            nn.Linear(256, num_tokens * cross_attn_dim),
        )
        self.fusion_alpha = nn.Parameter(torch.tensor(0.5))
        self.to(device)
        print(f'✅ IdentityProjector: {arcface_dim}-d → {num_tokens}×{cross_attn_dim}-d')

    def forward(self, arcface_emb, clip_id_feat, lambda_id=1.0):
        B        = arcface_emb.shape[0]
        # clip_id_feat may be on CPU — move to same device
        clip_id_feat = clip_id_feat.to(arcface_emb.device)
        arc_tok  = self.arc_proj(arcface_emb).view(B, self.num_tokens, self.cross_attn_dim)
        clip_tok = self.clip_proj(clip_id_feat).view(B, self.num_tokens, self.cross_attn_dim)
        alpha    = torch.sigmoid(self.fusion_alpha)
        tokens   = F.normalize(alpha * arc_tok + (1-alpha) * clip_tok, dim=-1)
        return tokens * lambda_id


id_projector = IdentityProjector()
print('Stage 2-A ready')
