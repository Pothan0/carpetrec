"""AI attribute extraction (Plan Phase 2).

Every attribute is derived with classical CV or the CLIP encoder we ALREADY load —
no separate VLM / captioning model.

- colour, palette       : classical (quantised dominant colours -> perceptual names)
- shape, aspect         : image/segmentation aspect-ratio heuristics
- pattern, material,    : CLIP zero-shot prompt-ensemble over a fixed ontology
  style, border, medallion

This module is the single home for feature extraction; ingest.py imports the colour
helpers from here (keeps the import graph acyclic: ingest -> attributes, never back).
"""

from __future__ import annotations

import colorsys

import numpy as np
from PIL import Image

# --------------------------------------------------------------------------- colour
def _rgb_to_name(rgb) -> str:
    """Map a single RGB triple to a coarse retail-style colour name."""
    r, g, b = (float(c) / 255.0 for c in rgb)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    h *= 360.0

    if s < 0.15:  # achromatic
        if v < 0.20:
            return "black"
        if v > 0.88:
            return "white"
        return "grey"
    if v < 0.22:
        return "black"
    if 20 <= h < 70 and v > 0.6 and s < 0.45:
        return "beige"
    if h < 15 or h >= 345:
        return "red"
    if h < 40:
        return "brown" if v < 0.55 else "orange"
    if h < 70:
        return "yellow"
    if h < 165:
        return "green"
    if h < 255:
        return "blue"
    if h < 300:
        return "purple"
    return "pink"


def derive_color(img: Image.Image) -> str:
    """Grayscale-aware dominant colour. Returns 'grayscale' for mono scans."""
    rgb = np.asarray(img.convert("RGB")).astype(np.float32)
    spread = rgb.max(axis=2) - rgb.min(axis=2)
    if float(spread.mean()) < 12.0:
        return "grayscale"
    small = np.asarray(img.convert("RGB").resize((64, 64))).reshape(-1, 3)
    quant = (small // 32) * 32 + 16
    colors, counts = np.unique(quant, axis=0, return_counts=True)
    return _rgb_to_name(colors[counts.argmax()])


def derive_palette(img: Image.Image, k: int = 3) -> list[str]:
    """Top-k distinct dominant colour names (most-frequent quantised colours)."""
    small = np.asarray(img.convert("RGB").resize((64, 64))).reshape(-1, 3)
    quant = (small // 32) * 32 + 16
    colors, counts = np.unique(quant, axis=0, return_counts=True)
    names: list[str] = []
    for idx in counts.argsort()[::-1]:
        name = _rgb_to_name(colors[idx])
        if name not in names:
            names.append(name)
        if len(names) >= k:
            break
    return names


def derive_shape_aspect(img: Image.Image) -> tuple[str, float]:
    """Coarse shape from aspect ratio (rugs in these scans fill the frame)."""
    w, h = img.size
    aspect = round(w / h, 2) if h else 1.0
    ratio = max(w, h) / max(1, min(w, h))
    if ratio < 1.2:
        shape = "square"
    elif ratio >= 1.8:
        shape = "runner"
    else:
        shape = "rectangle"
    return shape, aspect


# ------------------------------------------------------------- CLIP zero-shot tagging
# Fixed ontology (segment-agnostic). Edit here to tune; kept in code for prototype simplicity.
ONTOLOGY: dict[str, list[str]] = {
    "pattern": ["floral", "geometric", "medallion", "abstract", "striped", "solid"],
    "material": ["wool", "silk", "cotton", "synthetic", "jute"],
    "style": ["persian", "oriental", "modern", "bohemian", "traditional", "tribal"],
}
_TEMPLATES = {
    "pattern": ["a photo of a {} patterned rug", "a carpet with a {} design", "a rug with {} motifs"],
    "material": ["a rug made of {}", "a {} carpet", "a rug woven from {}"],
    "style": ["a {} style rug", "a {} carpet", "a photo of a {} rug"],
}
BINARY: dict[str, tuple[str, str]] = {
    # attribute -> (prompt_for_yes, prompt_for_no)
    "has_border": ("a rug with a decorative border", "a rug without any border"),
    "has_medallion": ("a rug with a central medallion", "a rug without a central medallion"),
}


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp((x - x.max()) * 100.0)  # temperature scaling (CLIP cosines are small-range)
    return e / e.sum()


class AttributeTagger:
    """Pre-embeds the ontology text prompts once; scores image vectors against them."""

    def __init__(self, clip_encoder, min_confidence: float = 0.0):
        self.clip = clip_encoder
        self.min_confidence = min_confidence
        self._cat: dict[str, tuple[list[str], np.ndarray]] = {}
        for attr, values in ONTOLOGY.items():
            mat = np.stack([self._text_vec(attr, v) for v in values]).astype("float32")
            self._cat[attr] = (values, mat)
        self._bin: dict[str, np.ndarray] = {
            attr: np.stack([self.clip.embed_text(yes), self.clip.embed_text(no)]).astype("float32")
            for attr, (yes, no) in BINARY.items()
        }

    def _text_vec(self, attr: str, value: str) -> np.ndarray:
        prompts = [t.format(value) for t in _TEMPLATES[attr]]
        vecs = np.stack([self.clip.embed_text(p) for p in prompts])
        mean = vecs.mean(axis=0)
        n = np.linalg.norm(mean)
        return mean / n if n else mean

    def tag_from_vec(self, clip_vec: np.ndarray) -> dict:
        """Return {pattern, material, style, has_border, has_medallion} for one image vector."""
        v = np.asarray(clip_vec, dtype="float32")
        out: dict[str, str] = {}
        for attr, (values, mat) in self._cat.items():
            probs = _softmax(mat @ v)
            i = int(probs.argmax())
            out[attr] = values[i] if probs[i] >= self.min_confidence else ""
        for attr, mat in self._bin.items():
            probs = _softmax(mat @ v)
            out[attr] = "yes" if probs[0] >= probs[1] else "no"
        return out
