import torch
import torch.nn as nn
 
from ..config import cfg

class IlluminationAdaIN(nn.Module):
    """Same as original, just smaller hidden dim (256→128)."""

    def __init__(self, sh_dim=27, device=DEVICE):
        super().__init__()
        self.sh_mlp = nn.Sequential(
            nn.Linear(sh_dim, 64), nn.SiLU(), nn.Linear(64, 128), nn.SiLU())
        channels = [1280, 640, 320, 320]
        self.gamma_heads = nn.ModuleList([nn.Linear(128, c) for c in channels])
        self.beta_heads  = nn.ModuleList([nn.Linear(128, c) for c in channels])
        for g, b in zip(self.gamma_heads, self.beta_heads):
            nn.init.zeros_(g.weight); nn.init.ones_(g.bias)
            nn.init.zeros_(b.weight); nn.init.zeros_(b.bias)
        self.to(device)
        print('✅ IlluminationAdaIN (lite) loaded')

    def forward(self, sh_coeffs, feature_map, res_idx, lambda_light=1.0):
        shared = self.sh_mlp(sh_coeffs)
        gamma  = (1.0 + self.gamma_heads[res_idx](shared)).view(*shared.shape[:-1], -1, 1, 1)
        beta   = self.beta_heads[res_idx](shared).view(*shared.shape[:-1], -1, 1, 1)
        mean   = feature_map.mean(dim=[2,3], keepdim=True)
        std    = feature_map.std(dim=[2,3],  keepdim=True).clamp(min=1e-5)
        f_out  = gamma * (feature_map - mean) / std + beta
        return (1 - lambda_light) * feature_map + lambda_light * f_out


illum_adain = IlluminationAdaIN()
print('Stage 2-C ready')
