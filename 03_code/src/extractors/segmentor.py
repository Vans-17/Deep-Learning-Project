class LiteFaceSegmentor:
    """
    Tries to load BiSeNet for accurate segmentation.
    Falls back gracefully to a fast landmark-based ellipse mask.
    The ellipse fallback is completely fine for a course project.
    """
    FACE_LABELS = [1,2,3,4,5,10,11,12,13]

    def __init__(self, device=DEVICE):
        self.device    = device
        self.net       = None
        self.available = False
        self._try_load_bisenet()

    def _try_load_bisenet(self):
        model_path = Path('bisenet_face_parsing.pth')
        if not model_path.exists():
            print('  Trying to download BiSeNet weights...')
            try:
                import subprocess
                subprocess.run(
                    ['gdown', '--fuzzy',
                     'https://drive.google.com/file/d/154JgKpzCPW82qINcVieuPH3fZ2e0P812/view?usp=sharing',
                     '-O', 'bisenet_face_parsing.pth'],
                    capture_output=True, timeout=60)
            except Exception:
                pass
        if model_path.exists():
            if not os.path.exists('face-parsing.PyTorch'):
                os.system('git clone https://github.com/zllrunning/face-parsing.PyTorch.git -q')
            sys.path.insert(0, 'face-parsing.PyTorch')
            try:
                from model import BiSeNet as _BiSeNet
                net = _BiSeNet(n_classes=19)
                net.load_state_dict(torch.load('bisenet_face_parsing.pth', map_location='cpu'))
                net.eval()
                # Keep BiSeNet on CPU to save VRAM — move to GPU only during inference
                self.net       = net
                self.available = True
                print('✅ BiSeNet loaded (CPU idle, GPU during inference)')
                return
            except Exception as e:
                print(f'  BiSeNet load failed ({e}) — using ellipse fallback')
        else:
            print('  BiSeNet unavailable — using fast ellipse mask (fine for coursework)')

    def _ellipse_mask(self, lmks_2d, H, W):
        mask = np.zeros((H, W), dtype=np.uint8)
        cx   = int(lmks_2d[:,0].mean())
        cy   = int(lmks_2d[:,1].mean())
        rx   = int((lmks_2d[:,0].max() - lmks_2d[:,0].min()) * 0.6)
        ry   = int((lmks_2d[:,1].max() - lmks_2d[:,1].min()) * 0.65)
        cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 1, -1)
        return mask.astype(bool)

    @torch.no_grad()
    def parse(self, image_rgb: np.ndarray, lmks_2d=None):
        H, W = image_rgb.shape[:2]
        if not self.available:
            fm = self._ellipse_mask(lmks_2d, H, W) if lmks_2d is not None \
                 else np.zeros((H, W), bool)
            return np.zeros((H,W), np.int32), fm, np.zeros((H,W), bool)

        # Move to GPU just for this forward pass
        net = self.net.to(self.device)
        tf  = T.Compose([T.ToPILImage(), T.Resize((512,512)), T.ToTensor(),
                         T.Normalize([.485,.456,.406],[.229,.224,.225])])
        x   = tf(image_rgb).unsqueeze(0).to(self.device)
        out = net(x)[0]
        seg = out.argmax(1).squeeze().cpu().numpy().astype(np.int32)
        seg = cv2.resize(seg.astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST)
        net.to('cpu')  # free VRAM immediately
        torch.cuda.empty_cache()
        face_mask = np.isin(seg, self.FACE_LABELS)
        hair_mask = (seg == 17)
        return seg.astype(np.int32), face_mask, hair_mask


bisenet = LiteFaceSegmentor()
print('Stage 1-C ready')
