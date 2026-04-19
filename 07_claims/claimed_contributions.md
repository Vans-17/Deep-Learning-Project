# Claimed Contributions

## What we reproduced
- We reproduced the core idea of the **LDFaceNet-based face manipulation pipeline**, including:
  - Identity feature extraction using ArcFace-style embeddings.
  - A latent diffusion framework with a VAE + UNet + scheduler setup.
  - Conditional generation using classifier-free guidance (CFG).
- We maintained the standard pipeline structure of:
  - Encoding input images into latent space.
  - Conditioning the diffusion process on identity and other attributes.
  - Decoding the final latent back into an image.
- We followed the original **LDFaceNet philosophy** of performing face manipulation without requiring full retraining.

---

## What we modified
- We redesigned the architecture into a **fully modular pipeline** with clear separation of components:
  - `extractors/`, `adaptors/`, `diffusion/`, and utility modules.
- We introduced **multi-source identity conditioning**:
  - Combined ArcFace embeddings (512-d) with CLIP-based identity features.
  - Built a custom `identity_projector` to map these into token space (2 × 768 tokens).
- We added a **pose-expression adaptor**:
  - Mapped pose, shape, and expression parameters into residuals injected at different UNet levels.
- We prepared an **illumination control module**:
  - Implemented AdaIN-based illumination conditioning using spherical harmonics (SH coefficients).
- We incorporated **latent blending utilities**:
  - Designed functions to blend source and target latents using segmentation masks.
- We structured the denoising process to support:
  - Residual injection at each step of the diffusion process.
- We added a **metrics module**:
  - Identity similarity, SSIM, L2 distance, and a custom disentanglement score.

---

## What did not work
- **BiSeNet-based face segmentation**:
  - Failed to produce reliable or consistent masks.
- **Face segmentor integration**:
  - Outputs were noisy or misaligned, reducing effectiveness in masking and blending.
- **Low-resolution outputs**:
  - Generated images lacked sharpness and fine-grained details.

---

## What we believe is our contribution
- A **clean, modular, and extensible pipeline design** for identity-conditioned diffusion:
  - Each component (identity, pose, illumination) is independently pluggable.
- A **novel identity fusion strategy**:
  - Combining ArcFace and CLIP features into a unified token representation.
- A **residual-based conditioning mechanism**:
  - Injecting pose and expression information directly into UNet layers.
- A **framework for disentangled control**:
  - Separate pathways for identity, pose, expression, and illumination.
- A **practical engineering contribution**:
  - Making the pipeline easier to debug, extend, and experiment with.
- A **foundation for future improvements**:
  - Illumination and inversion modules are already integrated structurally.

---

## Comparison with LDFaceNet

| Component / Idea | LDFaceNet (Original) | Our Pipeline |
|---|---|---|
| **Core Architecture** | Latent diffusion with guided denoising | Same base, but modularized |
| **Conditioning Signals** | Identity + segmentation | Identity + pose + expression + illumination |
| **Identity Representation** | Single embedding | ArcFace + CLIP fusion |
| **Identity Injection** | Direct conditioning | Token-based via identity projector |
| **Segmentation Usage** | Core component, works reliably | Used for latent blending (unstable) |
| **Pose / Expression Control** | Implicit | Explicit adaptor with residual injection |
| **Illumination Control** | Not modeled | AdaIN-based module (SH coefficients) |
| **Conditioning Mechanism** | Guided denoising | CFG + residual injection |
| **Training Requirement** | No retraining required | Mostly inference-time, similar intent |
| **Pipeline Structure** | Monolithic | Fully modular |
| **Latent Operations** | Standard diffusion | Added latent blending |
| **Inversion** | Not emphasized | DDIM inversion implemented (unstable) |
| **Metrics** | Visual evaluation | Identity, SSIM, L2, disentanglement |
| **Extensibility** | Limited | High |
| **Practical Issues** | Not emphasized | Segmentation failure, low resolution |

---

## Summary
While LDFaceNet uses a guided latent diffusion model conditioned primarily on identity and segmentation, our work extends this into a modular, multi-factor conditioning framework with explicit disentanglement of identity, pose, expression, and illumination. This improves flexibility and control, although current limitations remain in segmentation reliability and output fidelity.
