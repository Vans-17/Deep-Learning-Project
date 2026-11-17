import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image
from src.pipeline import init_models, run_warp_swap, run_pipeline
from src.utils import load_image
from src.metrics import compute_metrics

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True)
    ap.add_argument('--target', required=True)
    ap.add_argument('--outdir', default=str(Path(__file__).resolve().parents[2] / '05_results'))
    args = ap.parse_args()

    models = init_models()
    src = load_image(args.source)
    tgt = load_image(args.target)
    print(f"src={src.shape} tgt={tgt.shape}")

    configs = [
        ("warp_global",         'warp',      dict(warp_method='global')),
        ("warp_piecewise",      'warp',      dict(warp_method='piecewise')),
        ("blend_pw_s0.2",       'blend',     dict(warp_method='piecewise', strength=0.2)),
        ("blend_pw_s0.45",      'blend',     dict(warp_method='piecewise', strength=0.45)),
        ("blend_pw_s0.65",      'blend',     dict(warp_method='piecewise', strength=0.65)),
        ("hybrid_pw_s0.2",      'hybrid',    dict(warp_method='piecewise', strength=0.2)),
        ("hybrid_pw_s0.45",     'hybrid',    dict(warp_method='piecewise', strength=0.45)),
        ("diffusion_s0.45",     'diffusion', dict(strength=0.45)),
    ]

    rows = []
    for name, method, kw in configs:
        print(f"\n── {name} ──")
        if method == 'warp':
            out = run_warp_swap(src, tgt, models=models, **kw)
        elif method == 'blend':
            swapped = run_warp_swap(src, tgt, models=models, verbose=False,
                                    warp_method=kw.get('warp_method', 'global'))
            diffused = run_pipeline(src, tgt, models=models,
                                    init_img=swapped['result'], verbose=False, **kw)
            import cv2
            d = diffused['result']
            s = cv2.resize(swapped['result'], (d.shape[1], d.shape[0]))
            mask = cv2.resize(swapped['face_mask'].astype(np.float32),
                              (d.shape[1], d.shape[0]))[..., None]
            res = (s.astype(np.float32) * mask + d.astype(np.float32) * (1 - mask))
            out = {'result': res.astype(np.uint8)}
        elif method == 'hybrid':
            swapped = run_warp_swap(src, tgt, models=models, verbose=False)
            out = run_pipeline(src, tgt, models=models, init_img=swapped['result'],
                               verbose=False, **kw)
        else:
            out = run_pipeline(src, tgt, models=models, verbose=False, **kw)

        Image.fromarray(out['result']).save(f"{args.outdir}/abl_{name}.png")
        m = compute_metrics(src, tgt, out['result'], models['arcface'])
        rows.append((name, m))
        print(f"  ID={m['id_sim']:.4f}  SSIM={m['ssim']:.4f}  L2={m['l2']:.1f}")

    print("\n=== ABLATION SUMMARY ===")
    print(f"{'config':<20} {'ID-sim':>8} {'SSIM':>8} {'L2↓':>8}")
    print("-" * 48)
    for name, m in rows:
        print(f"{name:<20} {m['id_sim']:>8.4f} {m['ssim']:>8.4f} {m['l2']:>8.1f}")

if __name__ == '__main__':
    main()
