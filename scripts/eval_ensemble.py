"""Phase 4b: DINOv2 + Marqo-FashionSigLIP score-fusion ensemble for image->image.

Embeds the catalogue + sampled query sets with both encoders, then fuses cosine-score
matrices (alpha*dino + (1-alpha)*marqo) and sweeps alpha, reporting recall@1/5/10 + mAP
per query set. Pure numpy fusion (fast); weights are already cached.

    python -m scripts.eval_ensemble
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from PIL import Image

from carpet_search.config import load_settings
from carpet_search.embedding import ClipEncoder, get_dino
from carpet_search.eval.run_eval import build_real_set
from carpet_search.eval.synth import build_synthetic_set

KS = (1, 5, 10)
SAMPLE = 120


def _sample(qs, n=SAMPLE, seed=42):
    if len(qs) <= n:
        return qs
    rng = np.random.default_rng(seed)
    return [qs[i] for i in sorted(rng.choice(len(qs), n, replace=False))]


def _batch(enc, imgs, bs=16):
    return np.vstack([enc.embed_images(imgs[i : i + bs]) for i in range(0, len(imgs), bs)]).astype("float32")


def _recall(scores: np.ndarray, gts: list[str], skus: list[str]) -> dict:
    order = np.argsort(-scores, axis=1)
    rec = {k: 0 for k in KS}; ap = 0.0
    for qi, gt in enumerate(gts):
        ranked = [skus[j] for j in order[qi]]
        gi = ranked.index(gt) if gt in ranked else None
        for k in KS:
            if gi is not None and gi < k:
                rec[k] += 1
        ap += (1.0 / (gi + 1)) if gi is not None else 0.0
    q = len(gts)
    return {**{f"R@{k}": rec[k] / q for k in KS}, "mAP": ap / q}


def _fmt(m):
    return f"R@1={m['R@1']:.3f} R@5={m['R@5']:.3f} R@10={m['R@10']:.3f} mAP={m['mAP']:.3f}"


def main() -> None:
    s = load_settings()
    idmap = pd.read_parquet(s.id_map_path)
    skus = idmap["sku"].tolist()
    cat_imgs = [Image.open(s.project_root / r["image_path"]).convert("RGB") for _, r in idmap.iterrows()]

    print("loading encoders + embedding catalogue (DINOv2 + Marqo)...", flush=True)
    dino = get_dino(s.models.dino_name, s.models.image_size)
    marqo = ClipEncoder("hf-hub:Marqo/marqo-fashionSigLIP", None)
    Dc = _batch(dino, cat_imgs)
    Mc = _batch(marqo, cat_imgs)

    qsets = {
        "synthetic": _sample(build_synthetic_set(s, profile="default")),
        "mobile": _sample(build_synthetic_set(s, profile="mobile")),
        "real": _sample(build_real_set(s)),
    }

    for setn, qs in qsets.items():
        qimgs = [q for q, _ in qs]
        gts = [g for _, g in qs]
        Dq = _batch(dino, qimgs)
        Mq = _batch(marqo, qimgs)
        dS, mS = Dq @ Dc.T, Mq @ Mc.T
        print(f"\n[{setn}]  (n={len(gts)})", flush=True)
        print(f"  dino-only        {_fmt(_recall(dS, gts, skus))}", flush=True)
        print(f"  marqo-only       {_fmt(_recall(mS, gts, skus))}", flush=True)
        for a in (0.3, 0.4, 0.5, 0.6, 0.7):
            print(f"  ensemble a={a:.1f}    {_fmt(_recall(a * dS + (1 - a) * mS, gts, skus))}", flush=True)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
