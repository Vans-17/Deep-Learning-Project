# 🎭 LDFaceNet-Lite + BFM 3D Face Map

**Course Project Edition** An optimized, geometrically-grounded face swapping and animation pipeline based on Latent Diffusion and the **Basel Face Model (BFM)**.

---

## 🛠 Project Description
This project implements a 3D-aware face swapping technique. By using a **56-dimensional BFM feature vector** (40-d shape, 10-d expression, 6-d pose), the model disentangles identity from geometry. 

**The "DiffSwap" Trick:** This pipeline performs high-fidelity transfer by blending the **Source Shape** coefficients with the **Target Pose/Expression**, ensuring the swapped face fits the target's orientation while maintaining the source's structural bone density.

---

## 💻 Environment Setup

### ⚠️ Strict Requirements
* **Python Version:** `3.11.9` (Libraries are strictly incompatible with other versions).
* **Hardware:** Optimized for **NVIDIA RTX 3050** (Laptop/Desktop) with 4–8 GB VRAM.
* **ONNX Runtime:** Requires an older version (`1.17.1`) for model compatibility.

### 📦 Dependency Versions

# Core Diffusion & Transformers
pip install diffusers==0.27.2 transformers==4.40.0 accelerate==0.29.3

# Face Analysis & Processing
pip install insightface==0.7.3 face-alignment==1.4.1 kornia==0.7.2 scipy==1.11.4

# Compatibility-specific ONNX
pip install onnxruntime-gpu==1.17.1
🚀 Execution Commands
1. Training & Fine-tuning
The PoseExpressionAdapter and IdentityProjector include trainable parameters. Training focuses on the Orthogonality Loss to ensure CLIP identity features don't bleed into pose features:

Python
# To train the disentangler projections:
loss = clip_disentangler.orthogonality_loss()
loss.backward()
2. Evaluation
You can evaluate the 3DMM extractor's accuracy by running the pose estimation validation:

Python
for path in list(FACES_DIR.glob("*.jpg"))[:5]:
    img = np.array(Image.open(path).convert("RGB"))
    attrs = tdmm.extract(img)
    print(f"Yaw: {attrs['pose'][0].item():+.1f}°")
3. Demo / Inference
Launch the interactive UI directly within the notebook:

Python
# Launch the GUI
display(ui)

# Or run via manual pipeline call:
pipeline_result = run_pipeline(
    source_img=src_img,    # Identity provider
    target_img=tgt_img,    # Pose/Expression provider
    lambda_shape=1.0,      # Weight for source 3D shape
    strength=0.02          # Injection weight
)
📉 Hardware Optimization
To run on 4GB VRAM without crashes, the project uses:

FP16 Precision: All latents and weights are processed in half-precision.

CPU Offloading: The CLIP backbone is moved to CPU when idle.

Memory Management:

Python
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"


