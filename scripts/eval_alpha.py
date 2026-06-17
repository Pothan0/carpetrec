"""Sweep the design<->colour alpha on the real (Persian variant) set, routing through the
full two-axis search pipeline (grayscale-DINO structure + LAB/EMD colour). Shows what the
slider does to recall. alpha=1.0 is pure design, 0.0 is pure colour.

    python -m scripts.eval_alpha --sample 120 --alphas 1.0,0.6,0.3,0.0
"""

from __future__ import annotations

import argparse

import numpy as np

from carpet_search.config import load_settings
from carpet_search.eval.run_eval import build_real_set, evaluate_via_search
from carpet_search.search import CarpetSearch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=120)
    ap.add_argument("--alphas", default="1.0,0.6,0.3,0.0")
    args = ap.parse_args()

    s = load_settings()
    search = CarpetSearch(s)
    qs = build_real_set(s)
    if args.sample and args.sample < len(qs):
        rng = np.random.default_rng(s.eval.synth_seed)
        idx = sorted(rng.choice(len(qs), size=args.sample, replace=False))
        qs = [qs[i] for i in idx]

    ks = tuple(s.eval.ks)
    print(f"alpha sweep on {len(qs)} real variant queries vs full {search.dino_index.ntotal}-rug "
          f"gallery (design=1.0 ... colour=0.0)")
    for a in [float(x) for x in args.alphas.split(",")]:
        m = evaluate_via_search(search, qs, ks, alpha=a)
        print(f"  alpha={a:.2f}  R@1={m['recall@1']:.3f}  R@5={m['recall@5']:.3f}  "
              f"R@10={m['recall@10']:.3f}  mAP={m['mAP']:.3f}")


if __name__ == "__main__":
    main()
