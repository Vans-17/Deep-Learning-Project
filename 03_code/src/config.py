import torch
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class Config:
    device: str = field(default_factory=lambda: 'cuda' if torch.cuda.is_available() else 'cpu')
    dtype: torch.dtype = field(default_factory=lambda: torch.float16 if torch.cuda.is_available() else torch.float32)

    img_size: int = 512
    latent_size: int = 64

    # ── Backward-compatibility uppercase aliases ──────────────────────────
    @property
    def DEVICE(self) -> str:
        return self.device

    @property
    def DTYPE(self) -> torch.dtype:
        return self.dtype

    @property
    def IMG_SIZE(self) -> int:
        return self.img_size

    @property
    def LATENT_SIZE(self) -> int:
        return self.latent_size

    # ── LDM ───────────────────────────────────────────────────────────────
    ldm_model_id: str = "CompVis/stable-diffusion-v1-4"
    ddim_steps_inversion: int = 30
    ddim_steps_denoise: int = 50
    cfg_scale: float = 7.0
    id_guidance_scale: float = 2.5

    # ── ArcFace ───────────────────────────────────────────────────────────
    arcface_model_name: str = "buffalo_sc"
    arcface_det_size: tuple = (320, 320)

    # ── BFM extractor ─────────────────────────────────────────────────────
    bfm_n_shape: int = 40
    bfm_n_expr: int = 10

    # ── CLIP ──────────────────────────────────────────────────────────────
    clip_model_id: str = "openai/clip-vit-base-patch32"
    clip_pose_dim: int = 32
    clip_expr_dim: int = 32
    clip_light_dim: int = 16
    clip_id_dim: int = 128

    # ── BiSeNet ───────────────────────────────────────────────────────────
    bisenet_weights_path: str = "bisenet_face_parsing.pth"
    bisenet_repo_url: str = (
        "https://github.com/zllrunning/face-parsing.PyTorch.git"
    )
    bisenet_gdrive_url: str = (
        "https://drive.google.com/file/d/"
        "154JgKpzCPW82qINcVieuPH3fZ2e0P812/view?usp=sharing"
    )
    bisenet_face_labels: tuple = (1, 2, 3, 4, 5, 10, 11, 12, 13)

    # ── Identity Projector ────────────────────────────────────────────────
    id_proj_num_tokens: int = 2
    id_proj_cross_attn_dim: int = 768

    # ── Pose+Expression Adapter ───────────────────────────────────────────
    adapter_input_dim: int = 56
    adapter_injection_scale: float = 0.02
    adapter_latent_channels: tuple = (1280, 640, 320, 320)

    # ── Illumination AdaIN ────────────────────────────────────────────────
    illum_sh_dim: int = 27

    # ── Pipeline defaults ─────────────────────────────────────────────────
    default_lambda_id: float = 2.5
    default_lambda_pose: float = 0.3
    default_lambda_shape: float = 1.2
    default_lambda_expr: float = 0.3
    default_lambda_light: float = 0.5
    default_strength: float = 0.65
    default_blend_hardness: float = 0.5

    # ── Pose safety clamps ────────────────────────────────────────────────
    max_yaw_deg: float = 45.0
    max_pitch_deg: float = 30.0

    # ── Paths ─────────────────────────────────────────────────────────────
    faces_dir: Path = Path("faces")
    checkpoint_path: Path = Path("ldface_bfm_ckpt.pt")
    result_path: Path = Path("ldface_bfm_result.png")
    ablation_path: Path = Path("ablation_study.png")

    def __post_init__(self):
        self.latent_size = self.img_size // 8

# Singleton instance
cfg = Config()
