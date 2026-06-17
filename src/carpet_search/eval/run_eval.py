"""Evaluation metrics + config comparison (spec section 6.8).

A query derived from SKU X is correct if X appears in the returned top-k.
Two query sets:
  - synthetic: warped/composited versions of catalogue images (synth.py)
  - real: the g/l/t variant images in lifestyle/ (same rug, different capture)
Query embedding is batched so a full run is tractable on CPU.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from PIL import Image
from rich.console import Console
from rich.table import Table

from ..config import Settings
from ..preprocess import prepare_query
from ..search import CarpetSearch
from .synth import build_synthetic_set

console = Console()


def build_real_set(settings: Settings) -> list[tuple[Image.Image, str]]:
    """Real query set: lifestyle/{sku}__{variant}.jpg, ground-truth sku encoded in name."""
    out: list[tuple[Image.Image, str]] = []
    for f in sorted(settings.paths.lifestyle_images.glob("*.jpg")):
        sku = f.stem.split("__")[0]
        try:
            out.append((Image.open(f).convert("RGB"), sku))
        except Exception:
            continue
    return out


def build_mobile_set(settings: Settings) -> list[tuple[Image.Image, str]]:
    """Real mobile-photo query set: data/eval/mobile/{sku}__mobileNN.jpg (user-supplied)."""
    d = settings.project_root / "data" / "eval" / "mobile"
    out: list[tuple[Image.Image, str]] = []
    if d.is_dir():
        for f in sorted(d.glob("*.jpg")):
            sku = f.stem.split("__")[0]
            try:
                out.append((Image.open(f).convert("RGB"), sku))
            except Exception:
                continue
    return out


def _batch_embed(encoder, imgs: list[Image.Image], batch_size: int = 16) -> np.ndarray:
    chunks = [encoder.embed_images(imgs[i : i + batch_size]) for i in range(0, len(imgs), batch_size)]
    return np.vstack(chunks).astype("float32")


def evaluate(search: CarpetSearch, query_set: list[tuple[Image.Image, str]],
             ks=(1, 5, 10)) -> dict:
    if not query_set:
        return {f"recall@{k}": 0.0 for k in ks} | {"mAP": 0.0, "n": 0}

    ks = tuple(ks)
    maxk = max(ks)
    prepared = [prepare_query(q, search.settings) for q, _ in query_set]
    vecs = _batch_embed(search.dino, prepared)

    skus_col = search.id_map["sku"].tolist()
    recall = {k: 0 for k in ks}
    ap_sum = 0.0
    for vec, (_, gt) in zip(vecs, query_set):
        scores, ids = search.dino_index.search(vec.reshape(1, -1), maxk)
        ranked = [skus_col[i] for i in ids[0] if i >= 0]
        for k in ks:
            if gt in ranked[:k]:
                recall[k] += 1
        # single relevant item -> AP = 1/rank of first hit (0 if absent)
        ap_sum += next((1.0 / r for r, s in enumerate(ranked, 1) if s == gt), 0.0)

    n = len(query_set)
    out = {f"recall@{k}": recall[k] / n for k in ks}
    out["mAP"] = ap_sum / n
    out["n"] = n
    return out


def _print_metrics(title: str, metrics: dict) -> None:
    table = Table(title=title)
    table.add_column("metric")
    table.add_column("value", justify="right")
    for key, val in metrics.items():
        table.add_row(key, f"{val:.3f}" if isinstance(val, float) else str(val))
    console.print(table)


def run_all(settings: Settings, which: str = "both") -> dict:
    search = CarpetSearch(settings)
    ks = tuple(settings.eval.ks)
    results = {}

    if which in ("synthetic", "both", "all"):
        profile = settings.eval.synth_profile
        console.print("[bold]Building synthetic query set...[/bold]")
        qs = build_synthetic_set(settings, profile=profile)
        console.print(f"  {len(qs)} synthetic queries ({profile}); evaluating...")
        results["synthetic"] = evaluate(search, qs, ks)
        _print_metrics(f"Synthetic queries ({profile})", results["synthetic"])

    if which in ("real", "both", "all"):
        console.print("[bold]Building real query set (g/l/t variants)...[/bold]")
        qs = build_real_set(settings)
        console.print(f"  {len(qs)} real queries; evaluating...")
        results["real"] = evaluate(search, qs, ks)
        _print_metrics("Real queries (lifestyle variant images)", results["real"])

    if which in ("mobile", "all"):
        qs = build_mobile_set(settings)
        if qs:
            console.print(f"[bold]Mobile query set: {len(qs)} real phone photos; evaluating...[/bold]")
            results["mobile"] = evaluate(search, qs, ks)
            _print_metrics("Mobile queries (real phone photos)", results["mobile"])
        else:
            console.print("[yellow]No real mobile photos in data/eval/mobile/; "
                          "using the synthetic-mobile profile instead.[/yellow]")
            qs = build_synthetic_set(settings, profile="mobile")
            results["mobile"] = evaluate(search, qs, ks)
            _print_metrics("Mobile queries (synthetic-mobile)", results["mobile"])

    return results


def evaluate_via_search(search: CarpetSearch, query_set: list[tuple[Image.Image, str]],
                        ks=(1, 5, 10), alpha: float | None = None) -> dict:
    """Like evaluate() but routes each query through search.search_by_image, so the full
    pipeline (preprocess + TTA + two-axis alpha blend) is exercised. `include_eval_only=True`
    so the hidden Persian gold gallery is searchable; `alpha` overrides the design/colour blend."""
    if not query_set:
        return {f"recall@{k}": 0.0 for k in ks} | {"mAP": 0.0, "n": 0}
    ks = tuple(ks)
    maxk = max(ks)
    recall = {k: 0 for k in ks}
    ap_sum = 0.0
    for q, gt in query_set:
        ranked = [r.sku for r in search.search_by_image(q, top_k=maxk, alpha=alpha,
                                                        include_eval_only=True)]
        for k in ks:
            if gt in ranked[:k]:
                recall[k] += 1
        ap_sum += next((1.0 / r for r, s in enumerate(ranked, 1) if s == gt), 0.0)
    n = len(query_set)
    return {f"recall@{k}": recall[k] / n for k in ks} | {"mAP": ap_sum / n, "n": n}


# Gated configs swept by `evaluate --compare`. Each maps flag-name -> value; everything
# not listed is reset OFF first, so each row is independent.
_COMPARE_CONFIGS = [
    ("baseline", {}),
    ("+clahe", {"use_clahe": True}),
    ("+seg(grabcut)", {"use_segmentation": True, "segmentation_backend": "grabcut"}),
    ("+seg(sam3)", {"use_segmentation": True, "segmentation_backend": "sam3"}),
    ("+rect(grabcut)", {"use_rectification": True, "segmentation_backend": "grabcut"}),
    ("+rect(sam3)", {"use_rectification": True, "segmentation_backend": "sam3"}),
    ("+seg+rect(sam3)", {"use_segmentation": True, "use_rectification": True, "segmentation_backend": "sam3"}),
    ("+rect(sam3)+tta", {"use_rectification": True, "segmentation_backend": "sam3", "use_tta": True}),
    ("+tta", {"use_tta": True}),
]

# Reset values between rows: string-typed segmentation_backend needs its own default, not False.
_COMPARE_DEFAULTS = {
    "use_white_balance": False, "use_clahe": False, "use_segmentation": False,
    "use_rectification": False, "use_tta": False, "use_reranking": False,
    "segmentation_backend": "none",
}


def compare_configs(settings: Settings, which: str = "mobile", sample: int = 120) -> pd.DataFrame:
    """Sweep gated robustness configs (white-balance/CLAHE/segmentation/TTA/re-rank) on a
    chosen image-query set, routing through the full search pipeline. Keep only what helps."""
    search = CarpetSearch(settings)
    ks = tuple(settings.eval.ks)

    if which == "synthetic":
        qs = build_synthetic_set(settings, profile=settings.eval.synth_profile)
        label = f"synthetic ({settings.eval.synth_profile})"
    elif which == "real":
        qs = build_real_set(settings)
        label = "real (g/l/t variants)"
    else:
        qs = build_mobile_set(settings) or build_synthetic_set(settings, profile="mobile")
        label = "mobile (real photos)" if build_mobile_set(settings) else "mobile (synthetic-mobile)"

    if 0 < sample < len(qs):
        rng = np.random.default_rng(settings.eval.synth_seed)
        idx = sorted(rng.choice(len(qs), size=sample, replace=False))
        qs = [qs[i] for i in idx]
    console.print(f"[bold]Config comparison on {len(qs)} {label} queries[/bold]")

    P, R = search.settings.preprocess, search.settings.retrieval
    flag_owners = {
        "use_white_balance": P, "use_clahe": P, "use_segmentation": P, "use_rectification": P,
        "segmentation_backend": P, "use_tta": R, "use_reranking": R,
    }
    rows = []
    for name, flags in _COMPARE_CONFIGS:
        for fname, owner in flag_owners.items():
            setattr(owner, fname, _COMPARE_DEFAULTS[fname])   # reset all (type-aware)
        for fname, val in flags.items():
            setattr(flag_owners[fname], fname, val)           # apply this config
        m = evaluate_via_search(search, qs, ks)
        rows.append({"config": name, **{k: round(v, 3) if isinstance(v, float) else v for k, v in m.items()}})
        console.print(f"  {name:26s} R@1={m['recall@1']:.3f} R@5={m['recall@5']:.3f} mAP={m['mAP']:.3f}")

    df = pd.DataFrame(rows)
    table = Table(title=f"Robustness sweep ({label})")
    for col in df.columns:
        table.add_column(str(col), justify="right" if col != "config" else "left")
    for _, r in df.iterrows():
        table.add_row(*[str(r[c]) for c in df.columns])
    console.print(table)
    best = df.sort_values("recall@1", ascending=False).iloc[0]["config"]
    console.print(f"[green]Best config by recall@1: {best}[/green]")
    return df
