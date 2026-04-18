# ── Add this helper cell before your ablation loop ───────────────────────
#ABLATION HELPER LOOP
from skimage.metrics import structural_similarity as ssim
import cv2

def compute_metrics(src_img, tgt_img, result_img):
    """Compute all three metrics for one result."""
    # 1. ID-sim — already have this
    try:
        src_id  = arcface.extract(src_img)
        res_id  = arcface.extract(result_img)
        id_sim  = arcface.cosine_sim(src_id, res_id)
    except:
        id_sim  = float('nan')

    # 2. SSIM — structural similarity to target (pose preservation)
    tgt_resized = cv2.resize(tgt_img,    (256, 256))
    res_resized = cv2.resize(result_img, (256, 256))
    ssim_score  = ssim(tgt_resized, res_resized, channel_axis=2, data_range=255)

    # 3. FID proxy — L2 distance in pixel space (lower = more realistic)
    l2 = float(np.mean((tgt_resized.astype(float) -
                         res_resized.astype(float)) ** 2))

    return {'id_sim': id_sim, 'ssim': ssim_score, 'l2': l2}

#ABLATION CONFIGS
ablation_configs = {
    # ── Original 5 ───────────────────────────────────────────────
    "1. Baseline (Full)"     : {'id': 1.0, 'pose': 1.0, 'expr': 1.0, 'shape': 1.0},
    "2. w/o Pose"            : {'id': 1.0, 'pose': 0.0, 'expr': 1.0, 'shape': 1.0},
    "3. w/o Expression"      : {'id': 1.0, 'pose': 1.0, 'expr': 0.0, 'shape': 1.0},
    "4. w/o Identity"        : {'id': 0.0, 'pose': 1.0, 'expr': 1.0, 'shape': 1.0},
    "5. w/o Pose+Expr"       : {'id': 1.0, 'pose': 0.0, 'expr': 0.0, 'shape': 1.0},
    # ── New additions ─────────────────────────────────────────────
    "6. w/o Shape"           : {'id': 1.0, 'pose': 1.0, 'expr': 1.0, 'shape': 0.0},
    "7. ID Only"             : {'id': 1.0, 'pose': 0.0, 'expr': 0.0, 'shape': 0.0},
    "8. Shape+ID Only"       : {'id': 1.0, 'pose': 0.0, 'expr': 0.0, 'shape': 1.0},
}
#ABLATION RUN + METRICS
# ── Run ablation + collect metrics ───────────────────────────────────────
results  = {}
metrics  = {}

for title, config in ablation_configs.items():
    print(f"\nRunning: {title}...")
    output = run_pipeline_v2(
        source_img   = src_img,
        target_img   = tgt_img,
        lambda_id    = config['id'],
        lambda_pose  = config['pose'],
        lambda_expr  = config['expr'],
        lambda_shape = config['shape'],
        lambda_light = 0.5,
        strength     = 0.65,
        num_steps    = 50,
        blend_hardness = 0.5,
        verbose      = False
    )
    results[title] = output['result']
    metrics[title] = compute_metrics(src_img, tgt_img, output['result'])

# ── Visual grid ───────────────────────────────────────────────────────────
n = len(results)
fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
for ax, (title, img) in zip(axes, results.items()):
    m = metrics[title]
    ax.imshow(img)
    ax.set_title(
        f"{title}\n"
        f"ID={m['id_sim']:.3f}\n"
        f"SSIM={m['ssim']:.3f}",
        fontsize=9, fontweight='bold'
    )
    ax.axis('off')
plt.suptitle("Ablation Study — LDFaceNet-Lite + BFM", fontweight='bold', fontsize=13)
plt.tight_layout()
plt.savefig('ablation_study.png', dpi=150, bbox_inches='tight')
plt.show()

# ── Print metrics table ───────────────────────────────────────────────────
print(f"\n{'Config':<25} {'ID-sim':>8} {'SSIM':>8} {'L2↓':>10}")
print("─" * 55)
for title, m in metrics.items():
    print(f"{title:<25} {m['id_sim']:>8.4f} {m['ssim']:>8.4f} {m['l2']:>10.1f}")


#3-PAIR AVERAGE
# ── Load 3 frontal pairs from LFW ────────────────────────────────────────
import random
from pathlib import Path
from PIL import Image
import numpy as np

FACES_DIR    = Path(r"C:\Users\pragn\Desktop\e\DL\face_env\lfw_flat")
MAX_YAW      = 40
MAX_ATTEMPTS = 500

all_imgs = list(FACES_DIR.glob("*.jpg"))
random.shuffle(all_imgs)

def pick_frontal(pool, exclude_paths=[], max_yaw=MAX_YAW):
    for path in pool:
        if path in exclude_paths:
            continue
        img = np.array(Image.open(path).convert("RGB"))
        try:
            yaw = tdmm.extract(img)['pose'][0].item()
            if abs(yaw) < max_yaw:
                return img, path
        except:
            continue
    return None, None

pool = all_imgs[:MAX_ATTEMPTS]
used = []
pairs_imgs = []

for i in range(3):
    src_img_i, src_path_i = pick_frontal(pool, exclude_paths=used)
    used.append(src_path_i)
    tgt_img_i, tgt_path_i = pick_frontal(pool, exclude_paths=used)
    used.append(tgt_path_i)

    if src_img_i is None or tgt_img_i is None:
        raise RuntimeError(f"Could not find enough frontal pairs — raise MAX_ATTEMPTS")

    pairs_imgs.append((src_img_i, tgt_img_i))
    print(f"✅ Pair {i+1}: {src_path_i.name}  →  {tgt_path_i.name}")

# Unpack for direct access
src_img_1, tgt_img_1 = pairs_imgs[0]
src_img_2, tgt_img_2 = pairs_imgs[1]
src_img_3, tgt_img_3 = pairs_imgs[2]

# Preview all 3 pairs
fig, axes = plt.subplots(3, 2, figsize=(6, 9))
for i, (src, tgt) in enumerate(pairs_imgs):
    axes[i][0].imshow(src); axes[i][0].set_title(f"Pair {i+1} Source"); axes[i][0].axis('off')
    axes[i][1].imshow(tgt); axes[i][1].set_title(f"Pair {i+1} Target"); axes[i][1].axis('off')
plt.suptitle("3 LFW Pairs for Ablation", fontweight='bold')
plt.tight_layout()
plt.show()

print("\n✅ All 3 pairs ready — run multi-pair ablation cell below.")

#ULTI-PAIR ABLATION
# ── Multi-pair ablation ───────────────────────────────────────────────────
PAIRS = [
    (src_img_1, tgt_img_1),
    (src_img_2, tgt_img_2),
    (src_img_3, tgt_img_3),
]

averaged = {title: {'id_sim': [], 'ssim': [], 'l2': []}
            for title in ablation_configs}

for pair_idx, (src, tgt) in enumerate(PAIRS):
    print(f"\n── Pair {pair_idx+1}/3 ──")
    for title, config in ablation_configs.items():
        output = run_pipeline_v2(src, tgt, verbose=False,
                                 lambda_id=config['id'],
                                 lambda_pose=config['pose'],
                                 lambda_expr=config['expr'],
                                 lambda_shape=config['shape'])
        m = compute_metrics(src, tgt, output['result'])
        for k in averaged[title]:
            averaged[title][k].append(m[k])

# Print averaged results
print(f"\n{'Config':<25} {'ID-sim':>8} {'SSIM':>8}")
print("─" * 45)
for title, vals in averaged.items():
    print(f"{title:<25} "
          f"{np.nanmean(vals['id_sim']):>8.4f} "
          f"{np.nanmean(vals['ssim']):>8.4f}")

#QUANTITATIVE METRICS
# ── Preset Comparison: "Change X Only" ───────────────────────────────────
presets = {
    "Change Pose Only"      : dict(lambda_id=2.5, lambda_pose=1.5,
                                    lambda_expr=0.0, lambda_shape=0.0),
    "Change Expression Only": dict(lambda_id=2.5, lambda_pose=0.0,
                                    lambda_expr=1.5, lambda_shape=0.0),
    "Keep Identity Fixed"   : dict(lambda_id=4.0, lambda_pose=0.3,
                                    lambda_expr=0.3, lambda_shape=1.5),
}

fig, axes = plt.subplots(1, len(presets) + 2, figsize=(16, 4))
axes[0].imshow(src_img); axes[0].set_title('Source\n(Identity)');  axes[0].axis('off')
axes[1].imshow(tgt_img); axes[1].set_title('Target\n(Pose/Expr)'); axes[1].axis('off')

for ax, (name, params) in zip(axes[2:], presets.items()):
    out = run_pipeline_v2(src_img, tgt_img, verbose=False, **params)
    ax.imshow(out['result'])
    ax.set_title(f"{name}\nID-sim={out['identity_sim']:.3f}", fontsize=9)
    ax.axis('off')

plt.suptitle('Explicit Attribute Control — LDFaceNet-Lite + BFM', fontweight='bold')
plt.tight_layout()
plt.savefig('preset_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

#QUALITATIVE METRICS
def disentanglement_score(src_img, tgt_img,
                           vary='pose',    # what you're changing
                           fix='id'):      # what should stay fixed
    """
    Measures how well the pipeline isolates one attribute.
    
    vary='pose'  → pose should change, identity should not
    vary='expr'  → expression should change, identity should not
    vary='id'    → identity should change, pose should not
    """
    # Baseline: full swap
    base = run_pipeline_v2(src_img, tgt_img, verbose=False)

    # Ablation: zero out the varied attribute
    if vary == 'pose':
        ablated = run_pipeline_v2(src_img, tgt_img,
                                   lambda_pose=0.0, verbose=False)
        # Score: ID-sim should be SAME, pose diff should be HIGH
        id_base    = base['identity_sim']
        id_ablated = ablated['identity_sim']
        isolation  = abs(id_base - id_ablated)   # lower = better isolated
        print(f"Pose isolation score : {isolation:.4f} (lower = pose doesn't leak into ID)")

    elif vary == 'expr':
        ablated = run_pipeline_v2(src_img, tgt_img,
                                   lambda_expr=0.0, verbose=False)
        id_base    = base['identity_sim']
        id_ablated = ablated['identity_sim']
        isolation  = abs(id_base - id_ablated)
        print(f"Expr isolation score : {isolation:.4f}")

    return isolation

# Run all three
pose_iso  = disentanglement_score(src_img, tgt_img, vary='pose')
expr_iso  = disentanglement_score(src_img, tgt_img, vary='expr')

print(f"\nSummary:")
print(f"  Pose disentanglement : {pose_iso:.4f}")
print(f"  Expr disentanglement : {expr_iso:.4f}")
print(f"  (Lower = attributes are better isolated)")
