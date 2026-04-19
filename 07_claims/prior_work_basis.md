# Prior Work Basis

This project builds upon several key research papers and models in the areas of face manipulation, generative models, and representation learning. Below is a summary of the core works studied and how each influenced our approach.

---

## 1. LDFaceNet (https://arxiv.org/abs/2408.02078)
- **Core Idea**: Uses latent diffusion for face swapping with identity and segmentation guidance, without requiring retraining.
- **Influence on our work**:
  - Served as the **primary baseline architecture**.
  - Inspired the use of **latent diffusion (VAE + UNet + scheduler)** for face manipulation.
  - Motivated **conditioning-based generation instead of direct pixel-space editing**.
  - Influenced our decision to design a **training-free / inference-time pipeline**.

---

## 2. DiffSwap (https://openaccess.thecvf.com/content/CVPR2023/papers/Zhao_DiffSwap_High-Fidelity_and_Controllable_Face_Swapping_via_3D-Aware_Masked_Diffusion_CVPR_2023_paper.pdf)
- **Core Idea**: Diffusion-based face swapping with strong identity preservation and high-quality synthesis.
- **Influence on our work**:
  - Reinforced the effectiveness of **diffusion models for realistic face swapping**.
  - Inspired improvements in **identity preservation during generation**.
  - Highlighted the importance of **better conditioning strategies**, leading us to explore multi-source identity inputs.

---

## 3. CycleGAN (https://noiselab.ucsd.edu/ECE228_2018/Reports/Report16.pdf)
- **Core Idea**: Unpaired image-to-image translation using cycle consistency loss.
- **Influence on our work**:
  - Provided foundational understanding of **image-to-image translation without paired data**.
  - Introduced the concept of **preserving structural consistency across domains**.
  - Helped motivate disentanglement between **identity and other attributes**.

---

## 4. StarGAN (https://openaccess.thecvf.com/content_CVPR_2020/papers/Choi_StarGAN_v2_Diverse_Image_Synthesis_for_Multiple_Domains_CVPR_2020_paper.pdf)
- **Core Idea**: Multi-domain image-to-image translation using a single unified model.
- **Influence on our work**:
  - Inspired the idea of **handling multiple attributes (pose, expression, illumination)** in one framework.
  - Encouraged a **unified conditioning approach** instead of separate models for each transformation.
  - Motivated the design of **modular adaptors for different attributes**.

---

## 5. ArcFace (https://arxiv.org/abs/1801.07698)
- **Core Idea**: Deep face recognition model using additive angular margin loss for highly discriminative embeddings.
- **Influence on our work**:
  - Used as the **primary identity feature extractor (512-d embeddings)**.
  - Ensured strong **identity preservation during generation**.
  - Formed the backbone of our **identity conditioning pipeline**.

---

## 6. CLIP (Contrastive Language–Image Pretraining) (https://www.sciencedirect.com/science/article/abs/pii/S0893608022004579)
- **Core Idea**: Learns joint image-text representations with strong semantic understanding.
- **Influence on our work**:
  - Used to extract **additional identity-related features**.
  - Enabled **multi-modal identity fusion** when combined with ArcFace.
  - Improved robustness of identity representation beyond traditional face embeddings.

---

## 7. 3D Morphable Models (BFM - Basel Face Model)(https://arxiv.org/abs/1709.08398)
- **Core Idea**: Parametric 3D face model capturing shape, expression, and pose.
- **Influence on our work**:
  - Provided structured features for **pose, shape, and expression decomposition**.
  - Enabled explicit **disentanglement of geometric attributes**.
  - Used as input to the **pose-expression adaptor module**.

---

## 8. AdaIN (Adaptive Instance Normalization)(https://arxiv.org/abs/1703.06868)
- **Core Idea**: Aligns feature statistics (mean and variance) to transfer style.
- **Influence on our work**:
  - Inspired the **illumination control module**.
  - Used to map spherical harmonics coefficients to **gamma/beta parameters**.
  - Enabled a pathway for **lighting-aware generation**.

---

## 9. DDIM (Denoising Diffusion Implicit Models) (https://crad.ict.ac.cn/cn/article/pdf/preview/10.7544/issn1000-1239.202440191.pdf)
- **Core Idea**: Deterministic diffusion sampling enabling inversion and faster generation.
- **Influence on our work**:
  - Implemented for **latent inversion capability**.
  - Intended to improve **reconstruction and editing control**.
  - Although not fully stable, it provides a foundation for future improvements.

---

## Summary
The project is primarily grounded in **LDFaceNet and diffusion-based face swapping methods**, while incorporating ideas from:
- **Generative models** (CycleGAN, StarGAN),
- **Representation learning** (ArcFace, CLIP),
- **3D face modeling** (BFM),
- and **style/control mechanisms** (AdaIN, DDIM).

Together, these works informed our design of a **modular, multi-condition, and disentangled face manipulation pipeline**.
