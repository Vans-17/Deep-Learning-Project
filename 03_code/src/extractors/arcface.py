import insightface
from insightface.app import FaceAnalysis

class ArcFaceIdentityExtractor:
    """
    Extracts a 512-d L2-normalised identity embedding.
    Uses buffalo_sc (small/fast) instead of buffalo_l.
    """
    def __init__(self, device=DEVICE):
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' \
                    else ['CPUExecutionProvider']
        # buffalo_sc = smaller model, still accurate enough for a course project
        self.app = FaceAnalysis(name='buffalo_sc', providers=providers)
        # Smaller detection size saves VRAM
        self.app.prepare(ctx_id=0 if device == 'cuda' else -1, det_size=(320, 320))
        self.device = device
        print('✅ ArcFace (buffalo_sc) loaded')

    @torch.no_grad()
    def extract(self, image: np.ndarray) -> torch.Tensor:
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        faces   = self.app.get(img_bgr)
        if len(faces) == 0:
            raise ValueError('No face detected!')
        face = sorted(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]),
                      reverse=True)[0]
        emb = torch.tensor(face.normed_embedding, dtype=torch.float32)
        return emb.unsqueeze(0).to(self.device)  # (1, 512)

    def cosine_sim(self, a: torch.Tensor, b: torch.Tensor) -> float:
        return F.cosine_similarity(a, b).item()


arcface = ArcFaceIdentityExtractor()
print('Stage 1-A ready')
