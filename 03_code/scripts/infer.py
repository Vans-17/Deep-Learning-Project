# Usage: python scripts/infer.py --source src.jpg --target tgt.jpg --output result.png

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import cv2
from PIL import Image
from src.pipeline import init_models, run_pipeline, run_warp_swap
from src.utils import load_image

def main():
    parser = argparse.ArgumentParser(description="LDFaceNet face swap")
    parser.add_argument('--source',       required=True,  help="Source face image (identity donor)")
    parser.add_argument('--target',       required=True,  help="Target face image (pose donor)")
    parser.add_argument('--output',       default='result.png')
    parser.add_argument('--method',       default='diffusion', choices=['warp', 'diffusion', 'hybrid', 'blend'],
                        help="warp: landmark-based swap (fast, reliable identity). "
                             "diffusion: LDM img2img only (identity conditioning is untrained and weak). "
                             "hybrid: warp swap seeds the diffusion pipeline (VAE → denoise → decode). "
                             "blend: warp swap + diffusion, then alpha-blend the identity face back on top.")
    parser.add_argument('--warp_method',  default='global', choices=['global', 'piecewise'],
                        help="global: single similarity transform (identity-preserving, may misalign mouth). "
                             "piecewise: Delaunay triangles aligned locally (exact fit, distorts identity more).")
    parser.add_argument('--warp_scale',   type=float, default=1.0,
                        help="Scale the source face before warping (e.g. 1.2 = 20%% larger on target)")
    parser.add_argument('--pw_weight',    type=float, default=0.0,
                        help="0..1: blend global warp with local piecewise alignment (0 = global only)")
    parser.add_argument('--warp_dy',      type=float, default=0.0,
                        help="Shift the swapped face vertically in px (positive = up)")
    parser.add_argument('--target_skin_weight', type=float, default=0.0,
                        help="0..1: influence of target skin tone over source (0=original, 0.8=strong target tone)")
    parser.add_argument('--skin_lightness',     type=float, default=0.0,
                        help="Lighten the swapped face skin tone (e.g. 0.3 for subtle lightening, 0.5 for visibly lighter/fairer skin)")
    parser.add_argument('--lambda_id',    type=float, default=15.0)
    parser.add_argument('--lambda_shape', type=float, default=2.0)
    parser.add_argument('--lambda_pose',  type=float, default=0.1)
    parser.add_argument('--lambda_expr',  type=float, default=0.1)
    parser.add_argument('--strength',     type=float, default=0.45)
    parser.add_argument('--cfg_scale',    type=float, default=None,
                        help="CFG guidance scale (default 7.0; lower = output stays closer to input)")
    parser.add_argument('--eta',          type=float, default=0.0,
                        help="DDIM stochasticity / sampling temperature (0=deterministic, 1=DDPM-like)")
    parser.add_argument('--diffusion_layers', type=int, default=1,
                        help="Number of chained diffusion refinement passes (blend/hybrid only)")
    args = parser.parse_args()

    print("Loading models...")
    models = init_models()

    src = load_image(args.source)
    tgt = load_image(args.target)

    print(f"Running pipeline ({args.method})...")
    warp_kw = dict(warp_method=args.warp_method,
                   warp_scale=args.warp_scale,
                   pw_weight=args.pw_weight,
                   warp_dy=args.warp_dy,
                   target_skin_weight=args.target_skin_weight,
                   skin_lightness=args.skin_lightness)
    if args.method == 'warp':
        output = run_warp_swap(src, tgt, models=models, **warp_kw)
    elif args.method == 'hybrid':
        swapped = run_warp_swap(src, tgt, models=models, verbose=False, **warp_kw)
        cur = swapped['result']
        for _ in range(max(1, args.diffusion_layers)):
            cur = run_pipeline(src, tgt, models=models, init_img=cur,
                               verbose=False, **vars(args))['result']
        output = dict(result=cur, identity_sim=None,
                      attributes=swapped['attributes'])
        try:
            src_id = models['arcface'].extract(src)
            res_id = models['arcface'].extract(cur)
            output['identity_sim'] = models['arcface'].cosine_sim(src_id, res_id)
        except Exception:
            pass
        print(f"  ID-sim (hybrid x{args.diffusion_layers}): {output['identity_sim']:.4f}")
    elif args.method == 'blend':
        swapped = run_warp_swap(src, tgt, models=models, verbose=False, **warp_kw)
        cur = swapped['result']
        for _ in range(max(1, args.diffusion_layers)):
            cur = run_pipeline(src, tgt, models=models, init_img=cur,
                               verbose=False, **vars(args))['result']
        d = cur
        s = cv2.resize(swapped['result'], (d.shape[1], d.shape[0]))
        mask = cv2.resize(swapped['face_mask'].astype(np.float32),
                          (d.shape[1], d.shape[0]))[..., None]
        result = (s.astype(np.float32) * mask
                  + d.astype(np.float32) * (1 - mask))
        output = dict(result=result.astype(np.uint8), identity_sim=None,
                      attributes=swapped['attributes'])
        try:
            src_id = models['arcface'].extract(src)
            res_id = models['arcface'].extract(output['result'])
            output['identity_sim'] = models['arcface'].cosine_sim(src_id, res_id)
        except Exception:
            pass
        print(f"  ID-sim (blend x{args.diffusion_layers}): {output['identity_sim']:.4f}")
    else:
        output = run_pipeline(src, tgt, models=models, **vars(args))

    Image.fromarray(output['result']).save(args.output)
    print(f"Saved → {args.output}")
    if output['identity_sim'] is not None:
        print(f"ID-sim : {output['identity_sim']:.4f}")

if __name__ == '__main__':
    main()
