# Usage: python scripts/eval.py --faces_dir 04_data/sample_inputs/ --n_pairs 3

import argparse
import csv
import numpy as np
from src.diffusion.ldm import load_ldm
from src.pipeline import run_pipeline
from src.metrics import compute_metrics
from src.utils import load_image, pick_frontal

ABLATION_CONFIGS = {
    "Full Pipeline"  : dict(lambda_id=4.0, lambda_pose=0.1, lambda_expr=0.1, lambda_shape=2.0),
    "w/o Shape"      : dict(lambda_id=4.0, lambda_pose=0.1, lambda_expr=0.1, lambda_shape=0.0),
    "w/o Pose"       : dict(lambda_id=4.0, lambda_pose=0.0, lambda_expr=0.1, lambda_shape=2.0),
    "w/o Expression" : dict(lambda_id=4.0, lambda_pose=0.1, lambda_expr=0.0, lambda_shape=2.0),
    "w/o Identity"   : dict(lambda_id=0.0, lambda_pose=0.1, lambda_expr=0.1, lambda_shape=2.0),
    "ID Only"        : dict(lambda_id=4.0, lambda_pose=0.0, lambda_expr=0.0, lambda_shape=0.0),
    "Shape+ID Only"  : dict(lambda_id=4.0, lambda_pose=0.0, lambda_expr=0.0, lambda_shape=2.0),
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--faces_dir', required=True)
    parser.add_argument('--n_pairs',   type=int, default=3)
    parser.add_argument('--main_csv',  default='05_results/main_results.csv')
    parser.add_argument('--ablation_csv', default='05_results/ablations.csv')
    args = parser.parse_args()

    models = load_ldm()
    pairs  = pick_n_frontal_pairs(args.faces_dir, args.n_pairs)  # from src/utils.py

    main_rows     = []
    ablation_rows = []

    for pair_idx, (src, tgt) in enumerate(pairs):
        print(f"\n── Pair {pair_idx+1}/{len(pairs)} ──")
        for config_name, lambdas in ABLATION_CONFIGS.items():
            out = run_pipeline(src, tgt, models, lambdas)
            m   = compute_metrics(src, tgt, out['result'], models['arcface'])
            row = dict(pair=pair_idx+1, config=config_name, **m)
            ablation_rows.append(row)
            if config_name == "Full Pipeline":
                main_rows.append(row)
            print(f"  {config_name:<20} ID={m['id_sim']:.3f}  SSIM={m['ssim']:.3f}")

    write_csv(main_rows,     args.main_csv)
    write_csv(ablation_rows, args.ablation_csv)
    print(f"\nSaved: {args.main_csv}")
    print(f"Saved: {args.ablation_csv}")

if __name__ == '__main__':
    main()
