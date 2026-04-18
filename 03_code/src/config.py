DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE  = torch.float16 if DEVICE == 'cuda' else torch.float32

print(f'✅ Device: {DEVICE} | dtype: {DTYPE}')
if DEVICE == 'cuda':
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'   GPU  : {torch.cuda.get_device_name(0)}')
    print(f'   VRAM : {total_vram:.1f} GB')
    if total_vram < 7.5:
        print('   ⚠️  < 8 GB VRAM detected — consider reducing IMG_SIZE to 192')
IMG_SIZE    = 512  # Recommended for your 4.3GB VRAM
LATENT_SIZE = IMG_SIZE // 8 
DTYPE       = torch.float16 # Halves memory usage
