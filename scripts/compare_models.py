"""Phase 4: embedding-model A/B for image->image similarity.

Compares the current DINOv2 against alternative pretrained encoders (SigLIP2,
Marqo-FashionSigLIP) by embedding the catalogue + sampled query sets with each, building
an in-memory FAISS index, and measuring recall@k / mAP on synthetic / mobile / real.

    python -m scripts.compare_models
"""

from __future__ import annotations

import faiss
import numpy as np
import pandas as pd
from PIL import Image

from carpet_search.config import load_settings
from carpet_search.embedding import ClipEncoder, get_dino
from carpet_search.eval.run_eval import build_mobile_set, build_real_set
from carpet_search.eval.synth import build_synthetic_set

SAMPLE = 120
KS = (1, 5, 10)


def _sample(qs, n=SAMPLE, seed=42):
    if len(qs) <= n:
        return qs
    rng = np.random.default_rng(seed)
    return [qs[i] for i in sorted(rng.choice(len(qs), n, replace=False))]


def _batch(enc, imgs, bs=16):
    return np.vstack([enc.embed_images(imgs[i : i + bs]) for i in range(0, len(imgs), bs)]).astype("float32")


def main() -> None:
    s = load_settings()
    idmap = pd.read_parquet(s.id_map_path)
    skus = idmap["sku"].tolist()

    print("loading catalogue images + query sets...", flush=True)
    cat_imgs = [Image.open(s.project_root / r["image_path"]).convert("RGB") for _, r in idmap.iterrows()]
    qsets = {
        "synthetic": _sample(build_synthetic_set(s, profile="default")),
        "mobile": _sample(build_synthetic_set(s, profile="mobile")),
        "real": _sample(build_real_set(s)),
    }

    def evaluate(index, enc):
        out = {}
        for name, qs in qsets.items():
            qv = _batch(enc, [q for q, _ in qs])
            _, I = index.search(qv, max(KS))
            rec = {k: 0 for k in KS}; ap = 0.0
            for row, (_, gt) in zip(I, qs):
                ranked = [skus[j] for j in row if j >= 0]
                for k in KS:
                    if gt in ranked[:k]:
                        rec[k] += 1
                ap += next((1.0 / r for r, sk in enumerate(ranked, 1) if sk == gt), 0.0)
            n = len(qs)
            out[name] = {**{f"R@{k}": rec[k] / n for k in KS}, "mAP": ap / n}
        return out

    candidates = [
        ("DINOv2 ViT-B/14 (current)", lambda: get_dino(s.models.dino_name, s.models.image_size)),
        ("SigLIP2 ViT-B/16", lambda: ClipEncoder("ViT-B-16-SigLIP2", "webli")),
        ("Marqo-FashionSigLIP", lambda: ClipEncoder("hf-hub:Marqo/marqo-fashionSigLIP", None)),
    ]

    results = {}
    for label, make in candidates:
        try:
            print(f"\n=== {label}: loading + embedding catalogue ===", flush=True)
            enc = make()
            cat = _batch(enc, cat_imgs)
            idx = faiss.IndexFlatIP(cat.shape[1])
            idx.add(cat)
            results[label] = evaluate(idx, enc)
            for setn, m in results[label].items():
                print(f"  {label} / {setn:9s}: R@1={m['R@1']:.3f} R@5={m['R@5']:.3f} mAP={m['mAP']:.3f}", flush=True)
        except Exception as exc:
            print(f"  ! {label} failed to load/run: {exc}", flush=True)

    print("\n================ EMBEDDING-MODEL A/B ================", flush=True)
    for setn in qsets:
        print(f"\n[{setn}]  (R@1 / R@5 / mAP)")
        for label in results:
            m = results[label][setn]
            print(f"  {label:28s} {m['R@1']:.3f} / {m['R@5']:.3f} / {m['mAP']:.3f}")
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
