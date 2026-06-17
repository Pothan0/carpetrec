"""Image preprocessing for queries.

Segmentation and rectification are GATED by config and fall back to passing the image through
unchanged whenever detection is unreliable, so the eval harness (`evaluate --compare`) can
measure whether they help. Two segmentation backends feed a SINGLE shared carpet mask
(`segmentation.carpet_mask`): "sam3" (default) and "grabcut"; with backend "none" the legacy
classical `segment_carpet` (GrabCut) and `rectify` (Canny quad) paths are used instead.

The mask is computed at most ONCE per query and reused by both rectification and background
removal. `prepare_query_with_meta` additionally reports what happened (for the UI before/after).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .rectify import _order_points, rectify_from_corners, rectify_from_mask


def to_pil(src) -> Image.Image:
    """Coerce a path / PIL image / numpy array (BGR or RGB) into an RGB PIL image."""
    if isinstance(src, Image.Image):
        return src.convert("RGB")
    if isinstance(src, (str, Path)):
        return Image.open(src).convert("RGB")
    if isinstance(src, np.ndarray):
        arr = src
        if arr.ndim == 3 and arr.shape[2] == 3:
            arr = arr[:, :, ::-1]  # assume OpenCV BGR -> RGB
        return Image.fromarray(arr.astype("uint8")).convert("RGB")
    raise TypeError(f"cannot convert {type(src)!r} to PIL.Image")


def segment_carpet(img: Image.Image, settings=None, mask=None) -> Image.Image:
    """Isolate the rug: white-out the background and crop to its bounding box.

    Uses the provided `mask` if given (so the dispatcher's single SAM3/GrabCut mask is reused);
    otherwise computes one via the configured backend (or classical GrabCut when no backend /
    settings). Passes the image through unchanged if the mask is empty/unreliable.
    """
    rgb = np.asarray(img.convert("RGB"))
    h, w = rgb.shape[:2]
    if h < 32 or w < 32:
        return img

    if mask is None:
        backend = getattr(getattr(settings, "preprocess", None), "segmentation_backend", "none")
        if settings is not None and backend != "none":
            from .segmentation import carpet_mask
            mask = carpet_mask(img, settings)
        else:
            from .segmentation import _grabcut_mask
            mask = _grabcut_mask(rgb)
    if mask is None:
        return img

    fg = (np.asarray(mask) > 0)
    if fg.shape != (h, w):
        fg = cv2.resize(fg.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
    if fg.sum() < 0.02 * fg.size:
        return img  # segmentation failed / negligible

    ys, xs = np.where(fg)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    out = rgb.copy()
    out[~fg] = 255  # white-out background
    crop = out[y0 : y1 + 1, x0 : x1 + 1]
    if crop.size == 0:
        return img
    return Image.fromarray(crop)


def rectify(img: Image.Image, settings=None) -> Image.Image:
    """Legacy classical 4-point rectification (Canny -> largest quad -> warp).

    Kept for `segmentation_backend: none`. The SAM3/GrabCut-mask path uses
    rectify.rectify_from_mask instead (far more robust + metric aspect via Zhang & He).
    """
    rgb = np.asarray(img.convert("RGB"))
    h, w = rgb.shape[:2]
    gray = cv2.GaussianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (5, 5), 0)
    edges = cv2.dilate(cv2.Canny(gray, 50, 150), np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img

    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 0.20 * h * w:
        return img  # largest contour too small to be the rug

    approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
    if len(approx) != 4:
        return img  # not a clean quadrilateral

    rect = _order_points(approx.reshape(4, 2).astype(np.float32))
    tl, tr, br, bl = rect
    out_w = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    out_h = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    if out_w < 10 or out_h < 10:
        return img

    dst = np.float32([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]])
    warped = cv2.warpPerspective(rgb, cv2.getPerspectiveTransform(rect, dst), (out_w, out_h))
    return Image.fromarray(warped)


def white_balance_grayworld(img: Image.Image) -> Image.Image:
    """Gray-world white balance: rescale channels so their means equalise (fixes colour cast)."""
    rgb = np.asarray(img.convert("RGB")).astype(np.float32)
    means = rgb.reshape(-1, 3).mean(axis=0)
    scale = means.mean() / np.clip(means, 1e-6, None)
    return Image.fromarray(np.clip(rgb * scale, 0, 255).astype(np.uint8))


def apply_clahe(img: Image.Image) -> Image.Image:
    """CLAHE local-contrast normalisation on luminance (preserves colour)."""
    bgr = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    merged = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    return Image.fromarray(cv2.cvtColor(merged, cv2.COLOR_BGR2RGB))


def prepare_query_with_meta(img, settings) -> tuple[Image.Image, dict]:
    """Apply the gated preprocessing steps; return (RGB image, meta).

    Order: colour/contrast normalisation -> segmentation backend mask (computed ONCE) ->
    rectify (warp to top-down) and/or segment (white-out + crop). meta describes what happened,
    for the UI before/after panel: {rectified, wh_ratio, backend, mask_found}.
    """
    P = settings.preprocess
    out = to_pil(img)
    meta = {"rectified": False, "wh_ratio": None,
            "backend": getattr(P, "segmentation_backend", "none"), "mask_found": False}

    if P.use_white_balance:
        out = white_balance_grayworld(out)
    if P.use_clahe:
        out = apply_clahe(out)

    if meta["backend"] == "none":
        # legacy classical paths (current behaviour for the eval baseline)
        if P.use_segmentation:
            out = segment_carpet(out, settings)
        if P.use_rectification:
            out = rectify(out, settings)
        return out, meta

    if meta["backend"] == "api":
        # Hosted SAM3 returns 4 corners (in the SENT image's frame); warp the SAME image locally.
        if P.use_rectification:
            from .segmentation import carpet_corners_api
            corners = carpet_corners_api(out, settings)
            meta["mask_found"] = corners is not None
            before = out
            out = rectify_from_corners(out, corners, settings)
            meta["rectified"] = out is not before
            if meta["rectified"]:
                meta["wh_ratio"] = round(out.size[0] / max(1, out.size[1]), 3)
        return out, meta

    if not (P.use_rectification or P.use_segmentation):
        return out, meta

    from .segmentation import carpet_mask
    mask = carpet_mask(out, settings)            # ONE SAM3/GrabCut call
    meta["mask_found"] = mask is not None

    if P.use_rectification:
        before = out
        out = rectify_from_mask(out, mask, settings)
        meta["rectified"] = out is not before    # rectify_from_mask returns the same object on passthrough
        if meta["rectified"]:
            meta["wh_ratio"] = round(out.size[0] / max(1, out.size[1]), 3)
        mask = None                               # stale after the warp changed the geometry

    if P.use_segmentation:
        if mask is None:
            mask = carpet_mask(out, settings)
        out = segment_carpet(out, settings, mask=mask)

    return out, meta


def prepare_query(img, settings) -> Image.Image:
    """Apply the gated preprocessing steps according to config, return an RGB image."""
    return prepare_query_with_meta(img, settings)[0]
