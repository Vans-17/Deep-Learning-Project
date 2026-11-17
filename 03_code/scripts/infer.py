# Usage: python scripts/infer.py --source src.jpg --target tgt.jpg --output result.png

import argparse
import numpy as np
from PIL import Image
from src.diffusion.ldm import load_ldm
from src.pipeline import run_pipeline
from src.utils import load_image

def main():
    parser = argparse.ArgumentParser(description="LDFaceNet face swap")
    parser.add_argument('--source',       required=True,  help="Source face image (identity donor)")
    parser.add_argument('--target',       required=True,  help="Target face image (pose donor)")
    parser.add_argument('--output',       default='result.png')
    parser.add_argument('--lambda_id',    type=float, default=4.0)
    parser.add_argument('--lambda_shape', type=float, default=2.0)
    parser.add_argument('--lambda_pose',  type=float, default=0.1)
    parser.add_argument('--lambda_expr',  type=float, default=0.1)
    parser.add_argument('--strength',     type=float, default=0.45)
    args = parser.parse_args()

    print("Loading models...")
    models = load_ldm()

    src = load_image(args.source)
    tgt = load_image(args.target)

    print("Running pipeline...")
    output = run_pipeline(src, tgt, models, vars(args))

    Image.fromarray(output['result']).save(args.output)
    print(f"Saved → {args.output}")
    print(f"ID-sim : {output['identity_sim']:.4f}")

if __name__ == '__main__':
    main()
