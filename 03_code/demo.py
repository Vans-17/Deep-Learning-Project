# Usage: python scripts/demo.py --source src.jpg --target tgt.jpg

import argparse
import ipywidgets as widgets
import matplotlib.pyplot as plt
from IPython.display import display
from src.diffusion.ldm import load_ldm
from src.pipeline import run_pipeline
from src.utils import load_image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--target', required=True)
    args = parser.parse_args()

    models  = load_ldm()
    src_img = load_image(args.source)
    tgt_img = load_image(args.target)

    # ── Sliders ──────────────────────────────────────────────────────
    s_id    = widgets.FloatSlider(value=4.0, min=0.0, max=6.0, step=0.1, description='Identity')
    s_shape = widgets.FloatSlider(value=2.0, min=0.0, max=4.0, step=0.1, description='Shape')
    s_pose  = widgets.FloatSlider(value=0.1, min=0.0, max=2.0, step=0.05, description='Pose')
    s_expr  = widgets.FloatSlider(value=0.1, min=0.0, max=2.0, step=0.05, description='Expression')
    run_btn = widgets.Button(description='▶ Run Swap', button_style='success')
    out_box = widgets.Output()

    def on_run(b):
        out_box.clear_output()
        with out_box:
            result = run_pipeline(src_img, tgt_img, models, dict(
                lambda_id=s_id.value, lambda_shape=s_shape.value,
                lambda_pose=s_pose.value, lambda_expr=s_expr.value,
            ))
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(src_img);         axes[0].set_title('Source');  axes[0].axis('off')
            axes[1].imshow(tgt_img);         axes[1].set_title('Target');  axes[1].axis('off')
            axes[2].imshow(result['result']); axes[2].set_title(f"Result  ID={result['identity_sim']:.3f}"); axes[2].axis('off')
            plt.tight_layout(); plt.show()

    run_btn.on_click(on_run)
    display(widgets.VBox([s_id, s_shape, s_pose, s_expr, run_btn, out_box]))

if __name__ == '__main__':
    main()
