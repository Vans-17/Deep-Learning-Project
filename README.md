# Deep-Learning-Project
Controlled Deepfake Generation: Attribute-Aware and Identity-Stable Face Synthesis
Problem Statement: Modern deepfake generators can swap faces or generate realistic portraits, but users have very limited control over what stays fixed and what is allowed to change. For example, changing pose or expression often unintentionally alters identity, skin tone, or facial attributes. This project focuses on building a controllable deepfake generation pipeline where identity, pose, expression, and lighting can be manipulated independently using existing GAN or diffusion-based models. The goal is not to build a new generator from scratch, but to add control knobs on top of pre-trained deepfake models so that generation becomes predictable, stable, and user guided.

Key Papers:

Dwij Mehta, Aditya Mehta, Pratik Narang, “LDFaceNet: Latent Diffusion-based Network for High-Fidelity Deepfake Generation,” arXiv, 2024. https://arxiv.org/abs/2408.02078

Wenliang Zhao et al., “DiffSwap: High-Fidelity and Controllable Face Swapping via 3D-Aware Masked Diffusion,” CVPR, 2023. https://openaccess.thecvf.com/content/CVPR2023/papers/Zhao_DiffSwap_High-Fidelity_and_Controllable_Face_Swapping_via_3D-Aware_Masked_Diffusion_CVPR_2023_paper.pdf

Yunjey Choi et al., “StarGAN v2: Diverse Image Synthesis for Multiple Domains,” CVPR, 2020.

https://openaccess.thecvf.com/content_CVPR_2020/papers/Choi_StarGAN_v2_Diverse_Image_Synthesis_for_Multiple_Domains_CVPR_2020_paper.pdf

Expected Research Outcome:

A user-controllable deepfake generation system where students demonstrate explicit controls such as “change expression only,” “change pose only,” or “keep identity fixed under variation.” The research contribution could be a simple latent-space constraint, disentanglement loss, or conditioning strategy that improves attribute isolation compared to baselines. Demo: an interactive UI where sliders control identity strength, pose, and expression, with side-by-side comparisons showing reduced identity drift and improved consistency.
