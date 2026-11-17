# Dependency set — base Lite + face3d for BFM 3DMM
!pip install -q diffusers==0.27.2 transformers==4.40.0 accelerate==0.29.3
!pip install -q insightface==0.7.3 onnxruntime-gpu==1.17.1
!pip install -q face-alignment==1.4.1
!pip install -q kornia==0.7.2
!pip install -q scipy==1.11.4

import sys
!{sys.executable} -m pip install onnxruntime

import os, sys, warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.transform import Rotation

from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    DDIMScheduler,
    StableDiffusionPipeline,
)
from transformers import CLIPProcessor, CLIPVisionModelWithProjection
import torchvision.transforms as T
import face_alignment

import torch
import gc
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
gc.collect()
torch.cuda.empty_cache()
