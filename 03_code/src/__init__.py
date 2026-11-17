# src/__init__.py
"""
LDFaceNet-Lite + BFM package init.
"""

import ssl
import warnings
warnings.filterwarnings('ignore')

# Bypass macOS missing CA root certificates for model downloads
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass

from src.config import cfg
from src.pipeline import init_models, run_pipeline

__all__ = [
    "cfg",
    "init_models",
    "run_pipeline",
]
