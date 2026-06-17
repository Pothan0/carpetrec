"""Image/text encoders (spec section 6.2).

- DINOv2 (768-d) for image->image similarity.
- CLIP via open_clip (512-d) for text->image similarity.

Both produce L2-normalised vectors so that FAISS inner-product == cosine similarity.
Models are loaded lazily/once and placed on the best available device.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch
from PIL import Image


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        n = np.linalg.norm(x)
        return x / n if n > 0 else x
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


class DinoEncoder:
    def __init__(self, name: str, image_size: int = 224, grayscale: bool = False):
        self.device = get_device()
        # DINOv2 input side length must be a multiple of 14.
        self.image_size = max(14, (image_size // 14) * 14)
        self.grayscale = grayscale
        self.model = torch.hub.load(
            "facebookresearch/dinov2", name, trust_repo=True
        ).to(self.device).eval()

        from torchvision import transforms

        ops = [transforms.Resize((self.image_size, self.image_size))]
        if grayscale:
            # Drop colour (chroma) but keep 3 channels: matching is then driven by
            # motif/structure/texture, not colour or gray-vs-colour tone.
            ops.append(transforms.Grayscale(num_output_channels=3))
        ops += [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        self.tf = transforms.Compose(ops)

    @torch.no_grad()
    def embed_images(self, imgs: list[Image.Image]) -> np.ndarray:
        batch = torch.stack([self.tf(im.convert("RGB")) for im in imgs]).to(self.device)
        feats = self.model(batch)  # CLS embedding, (B, 768)
        return _l2_normalize(feats.float().cpu().numpy())

    def embed_image(self, img: Image.Image) -> np.ndarray:
        return self.embed_images([img])[0]


class ClipEncoder:
    def __init__(self, name: str, pretrained: str):
        import open_clip

        self.device = get_device()
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            name, pretrained=pretrained
        )
        self.model = self.model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer(name)

    @torch.no_grad()
    def embed_images(self, imgs: list[Image.Image]) -> np.ndarray:
        batch = torch.stack([self.preprocess(im.convert("RGB")) for im in imgs]).to(self.device)
        feats = self.model.encode_image(batch)
        return _l2_normalize(feats.float().cpu().numpy())

    def embed_image(self, img: Image.Image) -> np.ndarray:
        return self.embed_images([img])[0]

    @torch.no_grad()
    def embed_text(self, text: str) -> np.ndarray:
        tokens = self.tokenizer([text]).to(self.device)
        feats = self.model.encode_text(tokens)
        return _l2_normalize(feats.float().cpu().numpy())[0]


# Cache encoders so the UI / search / eval share one loaded copy per process.
@lru_cache(maxsize=2)
def get_dino(name: str, image_size: int, grayscale: bool = False) -> DinoEncoder:
    return DinoEncoder(name, image_size, grayscale)


@lru_cache(maxsize=1)
def get_clip(name: str, pretrained: str) -> ClipEncoder:
    return ClipEncoder(name, pretrained)


@lru_cache(maxsize=1)
def get_marqo(name: str) -> ClipEncoder:
    """Second image encoder for the ensemble (e.g. hf-hub:Marqo/marqo-fashionSigLIP)."""
    return ClipEncoder(name, None)
