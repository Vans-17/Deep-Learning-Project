import torch
import torch.nn as nn
 
from ..config import cfg
 
class PoseExpressionAdapter(nn.Module):
    """
    BFM-upgraded adapter. Input is now:
        pose(6) + shape_coeff(40) + expr_coeff(10) = 56-d

    Key change vs. Lite: shape_coeff carries SOURCE identity geometry,
    while pose and expr come from TARGET — this is exactly the DiffSwap
    conditioning strategy: transfer identity shape into the target's animation.

    Injection weight raised from 0.003 → 0.02 because BFM coefficients are
    properly normalised (zero-mean, unit-variance in BFM PCA space), so the
    signal is no longer drowned by the clamp.
    """
    LATENT_CHANNELS = [1280, 640, 320, 320]
    INPUT_DIM       = 56   # pose(6) + shape(40) + expr(10)
    INJECTION_SCALE = 0.02 # was 0.003 — safe to raise with proper BFM coefficients

    def __init__(self, device=DEVICE):
        super().__init__()
        self.coeff_mlp = nn.Sequential(
            nn.Linear(self.INPUT_DIM, 128),
            nn.SiLU(),
            nn.Linear(128, 256),
            nn.SiLU(),
            nn.Linear(256, 512),
        )
        self.res_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(512, ch), nn.Tanh())
            for ch in self.LATENT_CHANNELS
        ])
        # Separate learned gates for each conditioning axis
        self.pose_gate  = nn.Parameter(torch.ones(1))
        self.shape_gate = nn.Parameter(torch.ones(1))   # NEW — identity shape gate
        self.expr_gate  = nn.Parameter(torch.ones(1))
        self.to(device)
        print(f'✅ PoseExpressionAdapter (BFM edition): '
              f'{self.INPUT_DIM}-d input, injection_scale={self.INJECTION_SCALE}')

    def forward(self, pose_6d, shape_40d, expr_10d,
                lambda_pose=1.0, lambda_shape=1.0, lambda_expr=1.0, **kwargs):
        """
        Args:
            pose_6d   : (B, 6)   — from TARGET image
            shape_40d : (B, 40)  — from SOURCE image  ← identity-preserving
            expr_10d  : (B, 10)  — from TARGET image
        """
        pose_s  = pose_6d   * lambda_pose  * torch.sigmoid(self.pose_gate)
        shape_s = shape_40d * lambda_shape * torch.sigmoid(self.shape_gate)
        expr_s  = expr_10d  * lambda_expr  * torch.sigmoid(self.expr_gate)

        feat      = self.coeff_mlp(torch.cat([pose_s, shape_s, expr_s], dim=-1))
        residuals = [h(feat) for h in self.res_heads]
        return {'residuals': residuals, 'heatmap_feat': feat}


pose_expr_adapter = PoseExpressionAdapter()
print('Stage 2-B ready')

