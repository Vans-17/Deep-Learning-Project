import torch
import torch.nn as nn
 
from ..config import cfg

class PoseExpressionAdapter(nn.Module):
    """
    BFM-upgraded adapter. Input is:
        pose(6) + shape_coeff(40) + expr_coeff(10) = 56-d

    Key change: shape_coeff carries SOURCE identity geometry,
    while pose and expr come from TARGET — DiffSwap conditioning strategy.
    """
    LATENT_CHANNELS = list(cfg.adapter_latent_channels)
    INPUT_DIM       = cfg.adapter_input_dim   # pose(6) + shape(40) + expr(10)
    INJECTION_SCALE = cfg.adapter_injection_scale

    def __init__(self, device=None):
        super().__init__()
        self.device = device or cfg.device
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
        # Learned gates for conditioning axes
        self.pose_gate  = nn.Parameter(torch.ones(1))
        self.shape_gate = nn.Parameter(torch.ones(1))
        self.expr_gate  = nn.Parameter(torch.ones(1))
        self.to(self.device)

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
