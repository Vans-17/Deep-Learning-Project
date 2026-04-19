# Claimed Contributions

## What we reproduced
- We reproduced the core idea of a **latent diffusion-based face manipulation pipeline inspired by LDFaceNet**, including:
  - A VAE + UNet + scheduler-based latent diffusion setup.
  - Identity-conditioned generation using face recognition embeddings.
  - Classifier-free guidance (CFG) for controlled denoising.
- We followed the original approach of performing **inference-time manipulation without full model retraining**.
- We implemented the standard pipeline flow:
  - Encode image → apply conditional denoising → decode to image.

---

## What we modified
- We restructured the pipeline into a **modular architecture**:
  - Organized into `extractors/`, `adaptors/`, and `diffusion/` modules.
- We experimented with **multi-source identity conditioning**:
  - Combined ArcFace embeddings with CLIP-based features.
  - Used an `identity_projector` to map them into conditioning tokens.
- We added a **pose-expression adaptor**:
  - Maps pose, shape, and expression parameters to residuals injected into the UNet.
- We implemented an **illumination control module (AdaIN-based)**:
  - Uses spherical harmonics coefficients to produce gamma/beta parameters.
  - This module is implemented but not fully integrated into the pipeline.
- We added **latent-space utilities**:
  - Latent blending using segmentation masks.
  - DDIM inversion (implemented but not used in the default pipeline).
- We included a **metrics module**:
  - Identity similarity, SSIM, L2 distance, and a basic disentanglement score.

---

## What did not work
- **Face segmentation (BiSeNet / segmentor)**:
  - Produced unreliable or misaligned masks.
  - Reduced effectiveness of latent blending.
- **Low output resolution**:
  - Generated images lacked fine details and sharpness.

---

## What we believe is our contribution
- The main contribution is **engineering and system design**, not a new algorithm:
  - A **modular and extensible pipeline** for diffusion-based face manipulation.
- We explored **multiple conditioning signals** (identity, pose, expression, illumination):
  - Demonstrated how they can be integrated into a single pipeline structure.
- We implemented **alternative conditioning strategies**:
  - Token-based identity projection.
  - Residual injection into the UNet.
- We provided a **framework for experimentation**:
  - Components can be modified or replaced independently.
- We also documented **practical limitations**:
  - Segmentation challenges
  - Resolution constraints
  - Instability in inversion and conditioning modules

---

## Comparison with LDFaceNet

| Component / Idea | LDFaceNet (Original) | Our Pipeline |
|---|---|---|
| **Core Architecture** | Latent diffusion with guided denoising | Same base, modularized |
| **Conditioning Signals** | Identity + segmentation | Identity + pose + expression (+ partial illumination) |
| **Identity Representation** | Single embedding | ArcFace + CLIP (experimental) |
| **Identity Injection** | Direct conditioning | Token-based projection |
| **Segmentation Usage** | Reliable and central | Implemented but not reliable |
| **Pose / Expression Control** | Implicit | Explicit adaptor (not fully validated) |
| **Illumination Control** | Not included | Implemented but not integrated |
| **Conditioning Mechanism** | Guided denoising | CFG + residual injection |
| **Training Requirement** | No retraining | Similar intent (inference-time) |
| **Pipeline Structure** | Monolithic | Modular |
| **Latent Operations** | Standard | Added blending (limited by segmentation) |
| **Inversion** | Not emphasized | Implemented but unstable |
| **Evaluation** | Visual quality | Added basic metrics |
| **Extensibility** | Limited | Higher (engineering design) |
| **Output Quality** | High and stable | Lower, inconsistent |

---

## Summary
This project is best understood as:
> An implementation and structured extension of a diffusion-based face manipulation pipeline, with a focus on modular design and exploratory conditioning mechanisms.

The primary contribution lies in:
- System design and modularization  
- Exploration of multiple conditioning inputs  
- Identification of practical challenges in implementation  

We do **not claim algorithmic novelty or superior performance** over existing methods.
