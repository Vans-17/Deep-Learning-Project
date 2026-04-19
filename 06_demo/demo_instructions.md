# Demo Instructions — LDFaceNet-BFM
**Target time: 3–5 minutes**  
**Run this end-to-end during viva to show the system working on real inputs.**

---

## Before the Viva

- [ ] Environment is created and all dependencies installed (see `README.md`)
- [ ] LFW images are in place at `04_data/sample_inputs/`
- [ ] Models have been downloaded at least once (first run downloads ~5 GB)
- [ ] Run the pipeline once as a dry run to confirm no errors
- [ ] Keep `05_results/ablations.csv` open in a tab to show quantitative results

---

## Step 1 — Activate Environment (30 seconds)

```bash
conda activate facenet
cd /path/to/project
```

Confirm GPU is available:
```bash
python -c "import torch; print(torch.cuda.get_device_name(0))"
# Expected: NVIDIA GeForce RTX 3050
```

---

## Step 2 — Run a Single Face Swap (2 minutes)

```bash
python scripts/infer.py \
  --source 04_data/sample_inputs/src_1.jpg \
  --target 04_data/sample_inputs/tgt_1.jpg \
  --output 06_demo/live_result.png
```

**Expected terminal output:**
```
Loading models...
Running pipeline...
━━ Stage 1: Attribute Extraction (BFM) ━━
━━ Stage 2: CLIP Disentangle + Project ━━
━━ Stage 3: VAE Encode ━━
━━ Stage 4: img2img Denoise ━━
━━ Stage 5: Blend + Decode ━━
✅ Done
Saved → 06_demo/live_result.png
ID-sim : 0.6xxx
```

Open `06_demo/live_result.png` and show the panel:
- **Left:** Source face (identity)
- **Middle:** Target face (pose/expression)
- **Right:** Result (source identity in target pose)

**Talk through:** *"The source person's identity has been transferred onto the target's head pose and expression. The background and lighting come from the target image."*

---

## Step 3 — Show Attribute Control (1 minute)

Run the same pair with shape conditioning turned off to show its importance:

```bash
python scripts/infer.py \
  --source 04_data/sample_inputs/src_1.jpg \
  --target 04_data/sample_inputs/tgt_1.jpg \
  --output 06_demo/no_shape.png \
  --lambda_shape 0.0
```

Compare `live_result.png` vs `no_shape.png` side by side.

**Talk through:** *"Without the BFM shape coefficients, identity preservation drops — you can see the face geometry shifts toward the target person. This confirms that shape conditioning is the key contribution over the baseline."*

Then point to `05_results/ablations.csv` and show the ID-sim drop in the `w/o Shape` row.

---

## Step 4 — Show Quantitative Results (30 seconds)

Open `05_results/ablations.csv` and highlight:

| What to point at | What to say |
|---|---|
| `Full Pipeline` row | Highest ID-sim across all configs |
| `w/o Shape` row | ID-sim drops significantly |
| `w/o Identity` row | ID-sim collapses to near zero — confirms ArcFace conditioning works |
| `ID Only` vs `Shape+ID Only` | Shape coefficients add measurable identity gain on top of ArcFace alone |

---

## Pre-Saved results for Reference

```
05_results/figures/ablation_study.png      ← ablation grid
05_results/figures/preset_comparison.png   ← attribute control comparison
06_demo/backup_result_pair1.png            ← pre-run output for pair 1
06_demo/backup_result_pair2.png            ← pre-run output for pair 2
```

> **Note:** Backup images are supplementary only. Always attempt the live run first.

---

## Quick Reference — Key Lambda Values

| Preset | `lambda_id` | `lambda_shape` | `lambda_pose` | `lambda_expr` |
|---|---|---|---|---|
| **Default (best identity)** | 4.0 | 2.0 | 0.1 | 0.1 |
| Pose only | 2.5 | 0.0 | 1.5 | 0.0 |
| Expression only | 2.5 | 0.0 | 0.0 | 1.5 |
| No shape (ablation) | 4.0 | 0.0 | 0.1 | 0.1 |
