"""Colour axis of the two-axis engine (design vs. colour).

Each catalogue image gets a compact COLOUR PROFILE: its dominant colours as KMeans-K
centroids in CIELAB space plus their pixel weights. At query time two profiles are
compared with the Earth Mover's Distance (optimal transport) using the perceptual
CIEDE2000 colour difference as the ground cost — so a query's teal is correctly judged
close to a catalogue rug's turquoise, not bin-matched.

Why this and not a colour histogram: EMD over a few perceptual centroids is robust to
lighting/whitebalance shifts and to a rug having one dominant + several accent colours,
which a fixed-bin histogram handles poorly.

Stored as color.npz (centroids (N,K,3) float32, weights (N,K) float32), aligned to the
SAME row order as id_map.parquet / dino.faiss so all axes index the same rugs.
""" 

from __future__ import annotations

import numpy as np
from PIL import Image

K = 5  # dominant colours per rug


def _center_crop(im: Image.Image, frac: float = 0.85) -> Image.Image:
    """Trim the outer margin so a white catalogue backdrop doesn't pollute the palette.

    Colour extraction is far more background-sensitive than the (grayscale) structure
    embedding, so we crop here even though we deliberately do NOT segment the structure
    path on clean catalogue shots (measured net-negative there)."""
    w, h = im.size
    cw, ch = max(1, int(w * frac)), max(1, int(h * frac))
    l, t = (w - cw) // 2, (h - ch) // 2
    return im.crop((l, t, l + cw, t + ch))


def color_profile(im: Image.Image, k: int = K, max_side: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """Return (centroids LAB (k,3) float32, weights (k,) float32 summing to 1) for one image."""
    from skimage.color import rgb2lab
    from sklearn.cluster import KMeans

    im = _center_crop(im.convert("RGB"))
    im = im.copy()
    im.thumbnail((max_side, max_side))
    arr = np.asarray(im, dtype=np.float32) / 255.0
    lab = rgb2lab(arr).reshape(-1, 3)
    km = KMeans(n_clusters=k, n_init=4, random_state=0).fit(lab)
    centroids = km.cluster_centers_.astype(np.float32)
    counts = np.bincount(km.labels_, minlength=k).astype(np.float32)
    total = counts.sum()
    weights = (counts / total).astype(np.float32) if total > 0 else np.full(k, 1.0 / k, np.float32)
    return centroids, weights


def build_color_index(settings) -> None:
    """Compute a colour profile per catalogue image, aligned to id_map row order."""
    import pandas as pd
    from tqdm import tqdm

    idmap = pd.read_parquet(settings.id_map_path)
    n = len(idmap)
    centroids = np.zeros((n, K, 3), dtype=np.float32)
    weights = np.zeros((n, K), dtype=np.float32)
    for i, row in tqdm(idmap.iterrows(), total=n, desc="colour profiles", unit="img"):
        path = settings.project_root / row["image_path"]
        with Image.open(path) as im:
            c, w = color_profile(im)
        centroids[i] = c
        weights[i] = w
    settings.color_index_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(settings.color_index_path, centroids=centroids, weights=weights)
    print(f"Built colour index: {n} profiles ({K} colours each) -> {settings.color_index_path}")


def load_color_index(settings) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(settings.color_index_path)
    return data["centroids"].astype(np.float32), data["weights"].astype(np.float32)


def profile_color_names(centroids: np.ndarray, weights: np.ndarray, top: int = 4):
    """Map a LAB colour profile to (primary_name, [palette_names]) via perceptual naming.

    Names come from the actual dominant pixels (not a retailer tag), ordered by total
    cluster weight. Primary is the heaviest single colour. This is what the storefront
    shows, so it stays consistent with the colour axis the EMD search actually uses."""
    from skimage.color import lab2rgb

    from .attributes import _rgb_to_name

    weight_by_name: dict[str, float] = {}
    primary = None
    for j in np.argsort(-np.asarray(weights)):
        rgb = lab2rgb(np.asarray(centroids[j], dtype=float)[None, None, :])[0, 0]
        name = _rgb_to_name((rgb * 255).astype(int))
        if primary is None:
            primary = name
        weight_by_name[name] = weight_by_name.get(name, 0.0) + float(weights[j])
    palette = sorted(weight_by_name, key=lambda n: -weight_by_name[n])[:top]
    return primary or "", palette


def color_distances(q_cent: np.ndarray, q_w: np.ndarray,
                    cents: np.ndarray, wts: np.ndarray) -> np.ndarray:
    """EMD (CIEDE2000 ground cost) from one query profile to all N catalogue profiles.

    Returns (N,) float32 distances; lower = more similar in colour.
    """
    import ot
    from skimage.color import deltaE_ciede2000

    q_cent = np.asarray(q_cent, dtype=np.float64)
    q_w = np.ascontiguousarray(q_w, dtype=np.float64)
    q_w = q_w / q_w.sum()
    n = cents.shape[0]
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        # (K,K) perceptual cost between every query colour and every catalogue colour.
        cost = deltaE_ciede2000(q_cent[:, None, :], cents[i][None, :, :].astype(np.float64))
        w_i = np.ascontiguousarray(wts[i], dtype=np.float64)
        w_i = w_i / w_i.sum()
        out[i] = ot.emd2(q_w, w_i, np.ascontiguousarray(cost))
    return out
