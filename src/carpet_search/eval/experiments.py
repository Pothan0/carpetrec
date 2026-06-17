"""A/B workbench (Plan Phase 0).

Formalises the throwaway `_bias_analysis.py` / `_gray_experiment.py` pattern: build an
in-memory FAISS index from a candidate encoder and compare retrieval metrics (via the
existing `run_eval.evaluate`) against one or more query sets, printing a delta table.
Used by Phase 3 (robustness) and Phase 4 (embedding A/B).
"""

from __future__ import annotations

import numpy as np
from rich.console import Console
from rich.table import Table

from .run_eval import evaluate

console = Console()


def build_inmemory_index(encoder, images, dim: int, batch_size: int = 16):
    """Embed `images` with `encoder` and return an in-memory IndexFlatIP (no persistence)."""
    import faiss

    chunks = [encoder.embed_images(images[i : i + batch_size]) for i in range(0, len(images), batch_size)]
    vecs = np.vstack(chunks).astype("float32")
    idx = faiss.IndexFlatIP(dim)
    idx.add(vecs)
    return idx


def evaluate_variant(search, encoder, cat_index, query_sets: dict, ks) -> dict:
    """Evaluate one (encoder, catalogue-index) variant across named query sets.

    Temporarily swaps `search.dino`/`search.dino_index` so the existing `evaluate()` runs
    unchanged, then restores them.
    """
    orig_enc, orig_idx = search.dino, search.dino_index
    search.dino, search.dino_index = encoder, cat_index
    try:
        return {name: evaluate(search, qs, ks) for name, qs in query_sets.items()}
    finally:
        search.dino, search.dino_index = orig_enc, orig_idx


def print_ab(title: str, named_metrics: dict, baseline: str | None = None) -> None:
    """named_metrics: {variant_name: {metric: value}}; optional baseline name → delta column."""
    table = Table(title=title)
    names = list(named_metrics)
    metric_keys = list(next(iter(named_metrics.values())))
    table.add_column("metric")
    for n in names:
        table.add_column(n, justify="right")
    if baseline and baseline in named_metrics:
        table.add_column("Δ best", justify="right")
    for mk in metric_keys:
        row = [mk]
        vals = []
        for n in names:
            v = named_metrics[n].get(mk)
            vals.append(v)
            row.append(f"{v:.3f}" if isinstance(v, float) else str(v))
        if baseline and baseline in named_metrics and isinstance(named_metrics[baseline].get(mk), float):
            floats = [v for v in vals if isinstance(v, float)]
            best = max(floats) if floats else 0.0
            row.append(f"{best - named_metrics[baseline][mk]:+.3f}")
        table.add_row(*row)
    console.print(table)
