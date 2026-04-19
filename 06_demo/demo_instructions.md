# Demo Instructions — LDFaceNet-BFM
**For the evaluating professor · Estimated demo time: 3–5 minutes**

---

## System Requirements

| Requirement | Specification |
|---|---|
| Python | 3.11.9 (strictly — other versions cause dependency conflicts) |
| GPU | NVIDIA RTX 3050 or equivalent |
| VRAM | Minimum 5 GB available |
| RAM | 8 GB system RAM recommended |
| OS | Windows 10/11 or Ubuntu 20.04+ |
| Storage | ~8 GB free (models + environment) |

---

## Environment Setup

### 1. Create and activate the conda environment

```bash
conda create -n facenet python=3.11.9
conda activate facenet
```

### 2. Install core dependencies

```bash
pip install diffusers==0.27.2
pip install transformers==4.40.0
pip install accelerate==0.29.3
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install opencv-python==4.9.0.80
pip install Pillow scipy kornia face-alignment scikit-image ipywidgets
pip install matplotlib notebook
```

### 3. Install InsightFace (not available via standard pip — requires special handling)

InsightFace cannot be installed directly on Windows via `pip install insightface` because
it requires compiled C++ extensions that are not shipped in the standard wheel.
Follow these steps exactly:

**Option A — Pre-compiled wheel (recommended for Windows):**
```bash
# Download the pre-compiled wheel from:
# https://github.com/Gourieff/Assets/raw/main/Insightface/insightface-0.7.3-cp311-cp311-win_amd64.whl
# Then install it:
pip install insightface-0.7.3-cp311-cp311-win_amd64.whl
```

**Option B — Linux / if Option A fails:**
```bash
pip install insightface==0.7.3
```

### 4. Install ONNX Runtime (required by InsightFace for face detection)

ONNX Runtime must match your CUDA version. For CUDA 12.x:

```bash
pip install onnxruntime-gpu==1.17.1
```

If you do not have a compatible GPU or encounter errors:
```bash
pip install onnxruntime==1.17.1   # CPU fallback — slower but functional
```

> **Note:** InsightFace will automatically download the `buffalo_sc` face detection model
> (~200 MB) on first run. An internet connection is required the first time only.
> Subsequent runs use the cached model at `~/.insightface/models/buffalo_sc/`.

### 5. Verify the installation

```bash
python -c "
import torch, insightface, onnxruntime, diffusers
print('torch     :', torch.__version__)
print('CUDA avail:', torch.cuda.is_available())
print('GPU       :', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')
print('VRAM (GB) :', round(torch.cuda.get_device_properties(0).total_memory/1e9,1) if torch.cuda.is_available() else 'n/a')
print('insightface:', insightface.__version__)
print('onnxruntime:', onnxruntime.__version__)
print('diffusers  :', diffusers.__version__)
"
```

**Expected output:**
```
torch     : 2.x.x+cu121
CUDA avail: True
GPU       : NVIDIA GeForce RTX 3050
VRAM (GB) : 8.0
insightface: 0.7.3
onnxruntime: 1.17.1
diffusers  : 0.27.2
```

---

## Running the Demo

### Step 1 — Open the notebook (30 seconds)

```bash
conda activate facenet
jupyter notebook LDFaceNet_BFM.ipynb
```

Run all cells from the top down to the **"Stage 1-A — ArcFace"** cell.
The first run will download the Stable Diffusion v1-4 weights (~4 GB) and
the `buffalo_sc` InsightFace model (~200 MB). This is a one-time download.
Subsequent runs load from cache and take approximately 2–3 minutes total.

---

### Step 2 — Load face images (30 seconds)

Run the **"Load Source & Target from Google Drive"** cell. This automatically:
- Mounts Google Drive
- Samples two frontal LFW face images (yaw < 40°)
- Displays them side by side

The source image donates **identity** (whose face appears in the output).
The target image donates **pose and expression** (the head angle and facial animation).

---

### Step 3 — Run the pipeline (approximately 2 minutes on RTX 3050)

Run the `run_pipeline_v2(...)` cell with default parameters:

```python
output = run_pipeline_v2(
    source_img   = src_img,
    target_img   = tgt_img,
    lambda_id    = 4.0,    # identity strength
    lambda_shape = 2.0,    # 3D face shape strength (key contribution)
    lambda_pose  = 0.1,    # head pose transfer
    lambda_expr  = 0.1,    # expression transfer
    strength     = 0.45,
    num_steps    = 50,
    verbose      = True,
)
```

The terminal will print progress through 5 stages. On completion, a three-panel
figure is displayed showing the source, target, and swapped result alongside
the ArcFace identity similarity score (ID-sim).

**What to observe in the result:**
- The source person's face geometry (jaw, cheekbones, eye spacing) is preserved
- The target's head orientation and expression are transferred
- The background and hair remain from the target image unchanged

---

### Step 4 — Ablation demonstration (1 minute)

To demonstrate the contribution of the BFM 3D shape coefficients,
run the same pipeline with `lambda_shape = 0.0`:

```python
output_no_shape = run_pipeline_v2(
    source_img   = src_img,
    target_img   = tgt_img,
    lambda_id    = 4.0,
    lambda_shape = 0.0,    # shape conditioning disabled
    lambda_pose  = 0.1,
    lambda_expr  = 0.1,
    strength     = 0.45,
    num_steps    = 50,
    verbose      = False,
)
print(f"ID-sim  (with shape)   : {output['identity_sim']:.4f}")
print(f"ID-sim  (without shape): {output_no_shape['identity_sim']:.4f}")
```

The drop in ID-sim score demonstrates that BFM shape conditioning contributes
meaningfully to identity preservation beyond what ArcFace cross-attention provides alone.

---

### Step 5 — Quantitative results (30 seconds)

Open `05_results/ablations.csv` to view the full ablation table.
Open `05_results/figures/ablation_components.png` to view the summary plot.

Key rows to note:

| Configuration | ID-sim | Interpretation |
|---|---|---|
| Full pipeline | highest | All components active |
| w/o Shape | lower | Face geometry lost without BFM |
| w/o Identity | near zero | ArcFace conditioning confirmed working |
| Shape + ID only | second highest | Shape is the dominant identity signal |

---

## Common Issues and Fixes

| Issue | Fix |
|---|---|
| `InsightFace install fails on Windows` | Use the pre-compiled `.whl` from Option A above |
| `onnxruntime-gpu not found` | Install CPU version: `pip install onnxruntime==1.17.1` |
| `CUDA out of memory` | Reduce `IMG_SIZE = 192` in Cell 3 and re-run from Cell 3 onward |
| `No face detected` | Image has extreme pose — re-run the image loader cell to sample a new pair |
| `buffalo_sc download hangs` | Check internet connection; model downloads to `~/.insightface/models/` |
| `ModuleNotFoundError: face3d` | Not required — the BFM extractor uses scipy only, no face3d needed |

---

## File Reference

```
LDFaceNet_BFM.ipynb          ← main notebook (run this)
04_data/sample_inputs/       ← pre-selected LFW image pairs
05_results/ablations.csv     ← full ablation results table
05_results/figures/          ← all plots and qualitative outputs
06_demo/                     ← backup result images if live run is skipped
requirements.txt             ← full dependency list with versions
