"""Carpet segmentation — SAM3 (text-prompted) with a classical GrabCut fallback.

`carpet_mask(img, settings)` is the single source of truth for the rug mask: BOTH the
background-removal path (`preprocess.segment_carpet`) and the perspective-rectification path
(`rectify.rectify_from_mask`) consume it, so the segmentation model runs at most ONCE per query.

Backends (config `preprocess.segmentation_backend`):
  - "sam3"    : facebook/sam3 via transformers, prompted with the word "carpet" (gated; needs
                HF_TOKEN). Falls back to a GrabCut mask if SAM3 is unauthenticated/unavailable.
  - "grabcut" : the classical GrabCut mask (no model, no token).
  - "none"    : no mask (callers pass the image through unchanged).

The SAM3 loader mirrors embedding.py: load once via @lru_cache, on the best device, eval+no_grad.
truststore is already injected in __init__.py so corporate-TLS downloads work; the gated model
only additionally needs a token, which is read from the environment and never hardcoded.
"""

from __future__ import annotations

import os
from functools import lru_cache

import cv2
import numpy as np
import torch
from PIL import Image

from .embedding import get_device


def _read_hf_token() -> str | None:
    """HF token for the gated SAM3 model — from the environment or the HF login cache
    (`huggingface-cli login`); never committed/hardcoded."""
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if tok:
        return tok
    try:
        from huggingface_hub import get_token
        return get_token()
    except Exception:
        return None


# ---------------------------------------------------------------- SAM3
class Sam3Segmenter:
    """Wraps facebook/sam3 for text-prompted ("carpet") instance segmentation."""

    def __init__(self, model_id: str):
        from transformers import Sam3Model, Sam3Processor

        self.device = get_device()
        token = _read_hf_token()
        if token is None:
            print(f"  ! HF_TOKEN not set; SAM3 ({model_id}) is gated and the download will fail. "
                  "Set HF_TOKEN (or run `huggingface-cli login`); search will fall back to GrabCut.")
        self.processor = Sam3Processor.from_pretrained(model_id, token=token)
        self.model = Sam3Model.from_pretrained(model_id, token=token).to(self.device).eval()

    @torch.no_grad()
    def mask(self, img: Image.Image, prompt: str = "carpet"):
        """Return one boolean carpet mask (uint8 0/255, native HxW) or None."""
        img = img.convert("RGB")
        w, h = img.size
        inputs = self.processor(images=img, text=prompt, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        sizes = inputs.get("original_sizes")
        target_sizes = sizes.tolist() if sizes is not None else [[h, w]]
        res = self.processor.post_process_instance_segmentation(
            outputs, threshold=0.5, mask_threshold=0.5, target_sizes=target_sizes)[0]
        masks = res.get("masks")
        if masks is None or len(masks) == 0:
            return None
        return _pick_best_mask(masks, res.get("scores"), h, w)


@lru_cache(maxsize=1)
def get_sam3(model_id: str) -> Sam3Segmenter:
    return Sam3Segmenter(model_id)


def _to_2d_uint8(m, h: int, w: int):
    """Coerce a SAM mask (torch/np, bool/float, possibly (1,H,W)) to a 0/255 HxW uint8 array."""
    if hasattr(m, "detach"):
        m = m.detach().cpu().numpy()
    m = np.squeeze(np.asarray(m))
    if m.ndim == 3:                      # (k,H,W) -> union
        m = m.any(axis=0) if m.shape[0] <= m.shape[-1] else m[..., 0]
    if m.ndim != 2:
        return None
    mb = (m > 0.5).astype(np.uint8) * 255
    if mb.shape != (h, w):
        mb = cv2.resize(mb, (w, h), interpolation=cv2.INTER_NEAREST)
    return mb


def _pick_best_mask(masks, scores, h: int, w: int):
    """Largest carpet instance (gently weighted by score); reject a whole-frame mask."""
    best, best_rank = None, 0.0
    for i, m in enumerate(masks):
        mb = _to_2d_uint8(m, h, w)
        if mb is None:
            continue
        area = int((mb > 0).sum())
        if area == 0 or area / (h * w) > 0.985:    # empty / whole-frame (likely floor or failure)
            continue
        score = float(scores[i]) if scores is not None and i < len(scores) else 1.0
        rank = area * (0.5 + 0.5 * max(0.0, min(1.0, score)))
        if rank > best_rank:
            best, best_rank = mb, rank
    return best


# ---------------------------------------------------------------- GrabCut (no model)
def _grabcut_mask(rgb: np.ndarray):
    """Classical GrabCut foreground mask (full-res uint8 0/255) or None. Seeds a central rect."""
    h, w = rgb.shape[:2]
    scale = 256.0 / max(h, w)
    small = cv2.resize(rgb, (max(1, int(w * scale)), max(1, int(h * scale)))) if scale < 1.0 else rgb.copy()
    sh, sw = small.shape[:2]
    mask = np.zeros((sh, sw), np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    ix, iy = int(sw * 0.08), int(sh * 0.08)
    rect = (ix, iy, max(1, sw - 2 * ix), max(1, sh - 2 * iy))
    try:
        cv2.grabCut(cv2.cvtColor(small, cv2.COLOR_RGB2BGR), mask, rect, bgd, fgd, 3, cv2.GC_INIT_WITH_RECT)
    except Exception:
        return None
    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    if fg.sum() < 0.05 * fg.size:
        return None
    return cv2.resize(fg, (w, h), interpolation=cv2.INTER_NEAREST)


# ---------------------------------------------------------------- dispatcher
def carpet_mask(img: Image.Image, settings):
    """Boolean carpet mask (uint8 HxW 0/255) at native resolution, or None (pass image through)."""
    backend = getattr(settings.preprocess, "segmentation_backend", "none")
    rgb = np.asarray(img.convert("RGB"))
    h, w = rgb.shape[:2]
    if h < 32 or w < 32:
        return None

    if backend == "sam3":
        m = None
        try:
            prompt = getattr(settings.models, "sam3_prompt", "carpet")
            m = get_sam3(settings.models.sam3_name).mask(img, prompt)
        except Exception as e:  # gated/unauthenticated/load/inference failure
            print(f"  ! SAM3 segmentation failed ({type(e).__name__}: {e})")
        if m is None and getattr(settings.preprocess, "sam3_fallback_grabcut", True):
            m = _grabcut_mask(rgb)
        return m

    if backend == "grabcut":
        return _grabcut_mask(rgb)
    return None


# ---------------------------------------------------------------- remote corner API
def carpet_corners_api(img: Image.Image, settings):
    """POST the image to a hosted SAM3 corner service and return 4 corners (4,2 float32) or None.

    The service (see notebooks/untitled13 / the Colab) accepts multipart `file` and returns
    {"corners": [[x,y]x4], "labels": [...]} in the SENT image's pixel frame — so we must warp
    the SAME image we send. Offloads the heavy SAM3 inference to a GPU box; the warp stays local.
    """
    url = (getattr(settings.models, "sam3_api_url", "") or "").strip()
    if not url:
        return None
    import io

    import httpx

    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=92)
    payload = buf.getvalue()
    try:
        r = httpx.post(
            url,
            files={"file": ("query.jpg", payload, "image/jpeg")},
            headers={"ngrok-skip-browser-warning": "true"},
            timeout=getattr(settings.models, "sam3_api_timeout", 60),
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"  ! SAM3 corner API failed ({type(e).__name__}: {e})")
        return None
    corners = data.get("corners") if isinstance(data, dict) else None
    if not corners or len(corners) != 4:
        print(f"  ! SAM3 corner API returned no/invalid corners: {str(data)[:200]}")
        return None
    return np.asarray(corners, dtype=np.float32).reshape(4, 2)
