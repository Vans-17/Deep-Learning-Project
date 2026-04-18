# ── Load in fp16 to halve VRAM ────────────────────────────────────────────────
LDM_MODEL_ID = 'CompVis/stable-diffusion-v1-4'

print(f'Loading LDM: {LDM_MODEL_ID} ...')
pipe = StableDiffusionPipeline.from_pretrained(
    LDM_MODEL_ID,
    torch_dtype=torch.float16,
    safety_checker=None,
    requires_safety_checker=False,
).to(DEVICE)

# Enable memory-efficient attention if xformers is available
try:
    pipe.enable_xformers_memory_efficient_attention()
    print('  xformers memory-efficient attention enabled ✅')
except Exception:
    # Not fatal; just uses standard attention
    print('  xformers not available — using standard attention')

vae       = pipe.vae.eval()
unet      = pipe.unet.eval()
scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
scheduler.set_timesteps(20)   # 50 → 20 steps
tokenizer = pipe.tokenizer
text_enc  = pipe.text_encoder

for model in [vae, unet, text_enc]:
    for p in model.parameters():
        p.requires_grad_(False)

with torch.no_grad():
    null_tok = tokenizer([''], return_tensors='pt', padding='max_length',
                         max_length=77, truncation=True).input_ids.to(DEVICE)
    NULL_EMB = text_enc(null_tok)[0]  # (1, 77, 768)

vram_used = torch.cuda.memory_allocated() / 1e9 if DEVICE == 'cuda' else 0
print(f'\n✅ LDM loaded | VRAM used so far: {vram_used:.2f} GB')
print(f'   U-Net params: {sum(p.numel() for p in unet.parameters())/1e6:.0f}M')
