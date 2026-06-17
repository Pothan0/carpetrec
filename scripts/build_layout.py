"""Build 2D layout coordinates + PCA basis for the spatial "Constellation/Atlas" view.

Two REAL projections per rug, from data we already have:
  - DESIGN: PCA(2) of the 768-d DINOv2 vectors (dino.faiss) — groups by pattern/structure.
  - COLOUR: PCA(2) of the weight-sorted LAB palette (color.npz) — groups by colour.
Each axis is percentile-stretched to [-1,1] (declusters the core while preserving local
distance) and sign-pinned (stable orientation across rebuilds). The PCA bases + clip
bounds are persisted so a live query can be projected into the SAME frame.

Writes web/constellation/layout.json (served by the existing /static mount) and
data/index/layout_basis.npz.

IMPORTANT: coords are joined to dino.faiss/color.npz/id_map by ROW ORDER only — re-run
this (and build_atlas) after ANY build_index / build_color_index / grayscale change.

    python -m scripts.build_layout
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from carpet_search.color_index import load_color_index
from carpet_search.config import load_settings


def _sign_pin(comp: np.ndarray) -> np.ndarray:
    """Force the largest-magnitude loading of each component positive (stable orientation)."""
    for k in range(comp.shape[0]):
        j = int(np.argmax(np.abs(comp[k])))
        if comp[k, j] < 0:
            comp[k] = -comp[k]
    return comp


def _spread(v: np.ndarray):
    """Robust 2/98-percentile stretch to [-1,1] per axis (keeps local distance, declusters core)."""
    lo = np.percentile(v, 2, axis=0)
    hi = np.percentile(v, 98, axis=0)
    rng = np.where(hi - lo == 0, 1.0, hi - lo)
    return np.clip((v - lo) / rng * 2 - 1, -1, 1), lo, hi


def main() -> None:
    import faiss
    from sklearn.decomposition import PCA

    s = load_settings()
    idmap = pd.read_parquet(s.id_map_path).reset_index(drop=True)
    n = len(idmap)

    # DESIGN projection
    dino = faiss.read_index(str(s.dino_index_path))
    X = dino.reconstruct_n(0, dino.ntotal).astype("float64")
    pca_d = PCA(n_components=2, random_state=0).fit(X)
    pca_d.components_ = _sign_pin(pca_d.components_)
    d2 = (X - pca_d.mean_) @ pca_d.components_.T
    print("design  evr:", pca_d.explained_variance_ratio_[:2].round(3))

    # COLOUR projection — weight-sorted palette so the feature is order-invariant
    cents, wts = load_color_index(s)                       # (n,5,3), (n,5)
    order = np.argsort(-wts, axis=1)
    rows = np.arange(n)[:, None]
    cs = cents[rows, order]                                # dominant colour first
    ws = wts[rows, order]
    feat = (cs * ws[..., None]).reshape(n, -1).astype("float64")   # (n,15)
    pca_c = PCA(n_components=2, random_state=0).fit(feat)
    pca_c.components_ = _sign_pin(pca_c.components_)
    c2 = (feat - pca_c.mean_) @ pca_c.components_.T
    print("colour  evr:", pca_c.explained_variance_ratio_[:2].round(3))

    d2s, d_lo, d_hi = _spread(d2)
    c2s, c_lo, c_hi = _spread(c2)

    ev = idmap["eval_only"].astype(str).str.strip().str.lower().eq("true").to_numpy()
    titles = idmap.get("title", pd.Series([""] * n)).astype(str).tolist()
    colors = idmap.get("color", pd.Series([""] * n)).astype(str).tolist()
    palettes = idmap.get("palette", pd.Series([""] * n)).astype(str).tolist()

    items = []
    for i, sku in enumerate(idmap["sku"].astype(str)):
        items.append({
            "sku": sku,
            "t": titles[i],
            "color": colors[i],
            "palette": palettes[i],
            "d": [round(float(d2s[i, 0]), 4), round(float(d2s[i, 1]), 4)],
            "c": [round(float(c2s[i, 0]), 4), round(float(c2s[i, 1]), 4)],
            "eval_only": bool(ev[i]),
        })
    out_dir = s.project_root / "web" / "constellation"
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump({"count": n, "items": items}, open(out_dir / "layout.json", "w"))
    np.savez(s.paths.index_dir / "layout_basis.npz",
             d_mean=pca_d.mean_, d_comp=pca_d.components_, d_lo=d_lo, d_hi=d_hi,
             c_mean=pca_c.mean_, c_comp=pca_c.components_, c_lo=c_lo, c_hi=c_hi)

    # verify
    cbox = float(np.mean((np.abs(c2s) < 0.3).all(axis=1)))
    dbox = float(np.mean((np.abs(d2s) < 0.3).all(axis=1)))
    print(f"wrote {out_dir / 'layout.json'} ({n} rugs) + layout_basis.npz")
    print(f"center-box frac (want colour < 0.25): design={dbox:.2f}  colour={cbox:.2f}")
    sku = idmap["sku"].astype(str)
    for name, mask in {"afshan": sku.str.startswith("afshan_"),
                       "lachak": sku.str.startswith("lachak"),
                       "revival": sku.str.startswith("revival_")}.items():
        if mask.any():
            m = mask.to_numpy()
            print(f"  {name:8s} design={d2s[m].mean(0).round(2)}  colour={c2s[m].mean(0).round(2)}")


if __name__ == "__main__":
    main()
