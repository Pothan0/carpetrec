"""Synthetic query generation (spec section 6.8).

Turns a clean catalogue image into a "user-style" photo of the SAME rug:
perspective warp + floor-background composite + colour-temperature shift + optional
occluder. Used to expose the query<->gallery domain gap during evaluation.
"""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from ..config import Settings


def _floor_background(w: int, h: int, rng: np.random.Generator) -> np.ndarray:
    base = float(rng.integers(80, 180))
    tone = np.array([base, base * 0.85, base * 0.7])  # warm, wood-ish
    bg = np.ones((h, w, 3), np.float32) * tone
    bg *= np.linspace(0.8, 1.1, h)[:, None, None]      # vertical light gradient
    bg += rng.normal(0, 8, (h, w, 3))
    return np.clip(bg, 0, 255).astype(np.uint8)


def _colour_temperature(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    t = rng.uniform(-1.0, 1.0)
    out = img.astype(np.float32)
    out[..., 0] *= 1.0 + 0.12 * t   # R
    out[..., 2] *= 1.0 - 0.12 * t   # B
    return np.clip(out, 0, 255).astype(np.uint8)


def _add_occluder(img: np.ndarray, rng: np.random.Generator) -> None:
    h, w = img.shape[:2]
    ow, oh = int(rng.integers(w // 6, w // 3)), int(rng.integers(h // 6, h // 3))
    x, y = int(rng.integers(0, w - ow)), int(rng.integers(0, h - oh))
    color = tuple(int(c) for c in rng.integers(0, 120, 3))
    cv2.rectangle(img, (x, y), (x + ow, y + oh), color, -1)


def make_synthetic_query(img: Image.Image, seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    rgb = np.asarray(img.convert("RGB"))
    h, w = rgb.shape[:2]

    # Shrink the rug so it sits within a larger "room" frame.
    scale = float(rng.uniform(0.55, 0.85))
    nw, nh = max(8, int(w * scale)), max(8, int(h * scale))
    rug = cv2.resize(rgb, (nw, nh))

    # Random perspective warp of the rug.
    src = np.float32([[0, 0], [nw, 0], [nw, nh], [0, nh]])
    jitter = 0.12
    dst = src + (rng.uniform(-jitter, jitter, src.shape) * np.array([nw, nh])).astype(np.float32)
    M = cv2.getPerspectiveTransform(src, dst.astype(np.float32))
    warped = cv2.warpPerspective(rug, M, (nw, nh))
    mask = cv2.warpPerspective(np.full((nh, nw), 255, np.uint8), M, (nw, nh))

    # Composite onto a procedural floor background.
    bg = _floor_background(w, h, rng)
    ox, oy = int(rng.integers(0, w - nw + 1)), int(rng.integers(0, h - nh + 1))
    roi = bg[oy : oy + nh, ox : ox + nw]
    roi[mask > 0] = warped[mask > 0]
    bg[oy : oy + nh, ox : ox + nw] = roi

    out = _colour_temperature(bg, rng)
    if rng.random() < 0.4:
        _add_occluder(out, rng)
    return Image.fromarray(out)


# ---- harder "mobile photo" degradations (Plan Phase 1b) --------------------------
def _glare(arr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h, w = arr.shape[:2]
    cx, cy = int(rng.integers(0, w)), int(rng.integers(0, h))
    r = int(rng.integers(min(h, w) // 6, max(2, min(h, w) // 3)))
    yy, xx = np.ogrid[:h, :w]
    mask = np.clip(1 - np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / r, 0, 1)[..., None]
    return np.clip(arr.astype(np.float32) + mask * float(rng.integers(120, 200)), 0, 255).astype(np.uint8)


def _motion_blur(arr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    k = int(rng.integers(3, 9))
    kernel = np.zeros((k, k), np.float32)
    kernel[k // 2, :] = 1.0 / k
    if rng.random() < 0.5:
        kernel = kernel.T
    return cv2.filter2D(arr, -1, kernel)


def _jpeg_roundtrip(arr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    q = int(rng.integers(25, 55))
    ok, enc = cv2.imencode(".jpg", cv2.cvtColor(arr, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
    if not ok:
        return arr
    return cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def make_mobile_query(img: Image.Image, seed: int) -> Image.Image:
    """Harder query: warp+floor+colour (from make_synthetic_query) + glare + motion blur + JPEG."""
    base = np.asarray(make_synthetic_query(img, seed))
    rng = np.random.default_rng(seed + 777)
    if rng.random() < 0.6:
        base = _glare(base, rng)
    if rng.random() < 0.6:
        base = _motion_blur(base, rng)
    base = _jpeg_roundtrip(base, rng)
    return Image.fromarray(base)


def build_synthetic_set(settings: Settings, n_per_sku: int | None = None,
                        profile: str = "default") -> list[tuple[Image.Image, str]]:
    df = pd.read_csv(settings.paths.metadata_csv)
    n = n_per_sku if n_per_sku is not None else settings.eval.synth_per_sku
    base_seed = settings.eval.synth_seed
    make = make_mobile_query if profile == "mobile" else make_synthetic_query
    out: list[tuple[Image.Image, str]] = []
    for i, row in df.iterrows():
        path = settings.project_root / row["image_path"]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            continue
        for j in range(n):
            out.append((make(img, seed=base_seed + i * 100 + j), str(row["sku"])))
    return out
