from pathlib import Path
import os
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision import transforms
from collections import OrderedDict
from tqdm import tqdm
from .model import BiT_M_R50x1
from typing import Union
from typing import Optional, Tuple, List
from torch import Tensor


@torch.no_grad()
def get_embedding(
    img: Union[str, Image.Image],
    model: torch.nn.Module,
    device: torch.device
) -> torch.Tensor:
    _preprocess = transforms.Compose([
        transforms.Resize(128, interpolation=Image.BILINEAR),
        transforms.Pad(  # Pad to square
            padding=lambda img: (
                (128 - img.width) // 2,
                (128 - img.height) // 2,
                (128 - img.width) - (128 - img.width) // 2,
                (128 - img.height) - (128 - img.height) // 2
            ),
            fill=255
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])

    if isinstance(img, str):
        with Image.open(img) as im:
            im = im.convert("RGB")
            tensor = _preprocess(im).unsqueeze(0).to(device)
    else:
        im = img.convert("RGB")
        tensor = _preprocess(im).unsqueeze(0).to(device)

    # 前向
    feats = model.features(tensor)
    if feats.ndim == 4:
        feats = feats.flatten(1)
    feats = F.normalize(feats, p=2, dim=1).squeeze(0)  # [D]

    return feats.cpu()


class LogoMatcher:
    def __init__(self, ref_dir: str | Path, model_weights_path: str | Path):
        self.ref_dir = Path(ref_dir)
        self.model_weights_path = Path(model_weights_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.ref_database: Optional[Tuple[List[Tensor], List[str]]] = None

    def __call__(self, query_logo: str, top_k: int = 5) -> list[tuple[str, float]]:
        self._initialize()
        # get aspect ratio
        with Image.open(query_logo) as q_img:
            ratio_q = q_img.width / q_img.height

        # Get embedding & normalize
        query_feat = get_embedding(query_logo, self.model, self.device)
        query_feat = query_feat.unsqueeze(0) # [1, D]

        # Stack & normalize all refs
        ref_feats, ref_paths = self.ref_database
        feats = torch.stack(ref_feats, dim=0)                  # [N, D]

        # Cosine sim
        sims = (feats @ query_feat.T).squeeze(1)             # [N]
        num_cand = min(len(sims), max(2*top_k, 50))
        top_vals, top_idxs = sims.topk(num_cand, largest=True)

        results = []
        for idx, score in zip(top_idxs.tolist(), top_vals.tolist()):
            path = ref_paths[idx]
            try:
                with Image.open(path) as c_img:
                    ratio_c = c_img.width / c_img.height
                if max(ratio_q, ratio_c) / min(ratio_q, ratio_c) <= 2.5: # aspect ratio check
                    results.append((path, score))
                    if len(results) >= top_k:
                        break
            except OSError as e:
                print(f"[WARN] Fail to open {path!r}: {e!r}")
                continue

        return results

    def _initialize(self):
        if self.model is None:
            # load model once
            model = BiT_M_R50x1(num_classes=277, zero_head=True)
            ckpt = torch.load(self.model_weights_path, map_location='cpu')
            weights = ckpt.get('model', ckpt)
            sd = OrderedDict()
            for k, v in weights.items():
                nk = k.split('module.', 1)[1] if k.startswith('module.') else k
                sd[nk] = v
            model.load_state_dict(sd)
            model.to(self.device).eval()
            self.model = model

        if self.ref_database is None:
            feats, paths = [], []
            for cls in tqdm(os.listdir(self.ref_dir), desc="Load Logo reference database"):
                if cls.startswith('.'): continue
                d = self.ref_dir / cls
                if not d.is_dir(): continue
                for fn in os.listdir(d):
                    if not fn.lower().endswith(('.png','.jpg','.jpeg')): continue
                    p = d / fn
                    try:
                        emb = get_embedding(str(p), self.model, self.device)
                        feats.append(emb)
                        paths.append(str(p))
                    except OSError as e:
                        print(f"[WARN] Cannot process {p!r}: {e!r}")
            self.ref_database = (feats, paths)

