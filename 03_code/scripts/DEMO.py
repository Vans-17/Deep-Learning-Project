#DEMO
# ── Interactive Demo Cell (fixed) ─────────────────────────────────────────
import ipywidgets as widgets
from IPython.display import display

# ── Sliders ───────────────────────────────────────────────────────────────
slider_id    = widgets.FloatSlider(value=2.5, min=0.0, max=5.0, step=0.1,
                                   description='Identity:',   continuous_update=False)
slider_pose  = widgets.FloatSlider(value=0.3, min=0.0, max=2.0, step=0.05,
                                   description='Pose:',       continuous_update=False)
slider_expr  = widgets.FloatSlider(value=0.3, min=0.0, max=2.0, step=0.05,
                                   description='Expression:', continuous_update=False)
slider_shape = widgets.FloatSlider(value=1.2, min=0.0, max=3.0, step=0.1,
                                   description='Shape:',      continuous_update=False)
slider_str   = widgets.FloatSlider(value=0.55, min=0.3, max=0.9, step=0.05,
                                   description='Strength:',   continuous_update=False)

# ── Preset buttons ────────────────────────────────────────────────────────
btn_pose = widgets.Button(description='Pose Only',       button_style='info')
btn_expr = widgets.Button(description='Expression Only', button_style='warning')
btn_id   = widgets.Button(description='Fix Identity',    button_style='danger')

def set_pose_only(b):
    slider_id.value=2.5; slider_pose.value=1.5
    slider_expr.value=0.0; slider_shape.value=0.0

def set_expr_only(b):
    slider_id.value=2.5; slider_pose.value=0.0
    slider_expr.value=1.5; slider_shape.value=0.0

def set_fix_id(b):
    slider_id.value=4.0; slider_pose.value=0.3
    slider_expr.value=0.3; slider_shape.value=1.5

btn_pose.on_click(set_pose_only)
btn_expr.on_click(set_expr_only)
btn_id.on_click(set_fix_id)

# ── Run button + output widget ────────────────────────────────────────────
run_btn    = widgets.Button(description='▶ Run Swap', button_style='success')
output_box = widgets.Output()   # renamed to avoid collision with 'output' variable

# ── UI layout ─────────────────────────────────────────────────────────────
ui = widgets.VBox([
    widgets.HTML("<h3>🎭 LDFaceNet Interactive Demo</h3>"),
    widgets.HTML("<b>Presets:</b>"),
    widgets.HBox([btn_pose, btn_expr, btn_id]),
    widgets.HTML("<b>Manual controls:</b>"),
    slider_id, slider_pose, slider_expr, slider_shape, slider_str,
    run_btn,
    output_box,
])

# ── Callback ──────────────────────────────────────────────────────────────
def on_run(b):
    output_box.clear_output()
    with output_box:
        print("⏳ Running pipeline...")
        try:
            pipeline_result = run_pipeline_v2(
                source_img   = src_img,
                target_img   = tgt_img,
                lambda_id    = slider_id.value,
                lambda_pose  = slider_pose.value,
                lambda_expr  = slider_expr.value,
                lambda_shape = slider_shape.value,
                strength     = slider_str.value,
                num_steps    = 30,
                verbose      = False,
            )
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(src_img);
            axes[0].set_title('Source'); axes[0].axis('off')
            axes[1].imshow(tgt_img)
            axes[1].set_title('Target'); axes[1].axis('off')
            axes[2].imshow(pipeline_result['result'])
            axes[2].set_title(f"Result  ID-sim={pipeline_result['identity_sim']:.3f}",
                              color='green'); axes[2].axis('off')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"❌ {e}")

run_btn.on_click(on_run)
display(ui)
