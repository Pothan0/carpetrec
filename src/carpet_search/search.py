"""Query pipeline (spec section 6.5).

Image search routes through the DINO index; text search through the CLIP index.
Metadata filtering is post-search: overfetch top_k * factor, filter id_map rows,
then trim to top_k (FAISS flat has no native filter; fine at prototype scale).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from .color_index import color_distances, color_profile, load_color_index
from .config import Settings
from .embedding import get_clip, get_dino, get_marqo
from .index import load_indices
from .preprocess import prepare_query
from .schema import SearchResult

_META_FIELDS = ["title", "color", "size", "material", "pattern", "shape", "price",
                "palette", "aspect", "has_border", "has_medallion", "style"]


def _is_eval_only(row: pd.Series) -> bool:
    """Hidden eval-gallery rows (e.g. the Persian set): in the index, never user-facing."""
    return str(row.get("eval_only", "")).strip().lower() == "true"


def _minmax(x: np.ndarray) -> np.ndarray:
    """Scale to [0,1] across the candidate set (so structure-sim and colour-sim are comparable)."""
    lo, hi = float(x.min()), float(x.max())
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def _matches(row: pd.Series, filters: dict | None) -> bool:
    if not filters:
        return True
    for key, value in filters.items():
        if value in (None, "", "Any", "any"):
            continue
        cell = row.get(key)
        if cell is None or str(cell).strip().lower() != str(value).strip().lower():
            return False
    return True


class CarpetSearch:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.dino_index, self.clip_index, self.id_map = load_indices(settings)
        self.dino = get_dino(settings.models.dino_name, settings.models.image_size,
                             settings.preprocess.grayscale)
        self.clip = get_clip(settings.models.clip_name, settings.models.clip_pretrained)

        # Two-axis engine (default): structure = grayscale DINO cosine, colour = LAB/EMD.
        # The full structure matrix is reconstructed once so we can score every rug per query.
        self.marqo = None
        self._dino_mat = self._marqo_mat = None
        self._color_cent = self._color_wts = None
        if settings.retrieval.two_axis:
            try:
                self._dino_mat = self.dino_index.reconstruct_n(0, self.dino_index.ntotal)
                self._color_cent, self._color_wts = load_color_index(settings)
            except Exception as exc:
                print(f"  ! two-axis disabled (could not load colour index): {exc}")
                self._color_cent = None
        elif settings.retrieval.ensemble:
            # Legacy DINOv2 + Marqo ensemble (Plan Phase 4), shelved behind two_axis.
            try:
                import faiss

                self.marqo = get_marqo(settings.models.marqo_name)
                mq = faiss.read_index(str(settings.marqo_index_path))
                self._marqo_mat = mq.reconstruct_n(0, mq.ntotal)
                self._dino_mat = self.dino_index.reconstruct_n(0, self.dino_index.ntotal)
            except Exception as exc:
                print(f"  ! ensemble disabled (could not load Marqo index): {exc}")
                self.marqo = None

    # ------------------------------------------------------------------ helpers
    def _row_to_result(self, idx: int, score: float) -> SearchResult:
        row = self.id_map.iloc[idx]
        meta = {f: (None if pd.isna(row.get(f)) else row.get(f)) for f in _META_FIELDS}
        img_path = self.settings.project_root / row["image_path"]
        return SearchResult(
            sku=str(row["sku"]),
            score=float(score),
            image_path=str(img_path),
            metadata=meta,
        )

    def _search_index(self, index, qvec: np.ndarray, top_k: int, filters: dict | None,
                      include_eval_only: bool = False) -> list[SearchResult]:
        overfetch = min(top_k * self.settings.search.overfetch_factor, index.ntotal)
        scores, ids = index.search(qvec.reshape(1, -1).astype("float32"), overfetch)
        results: list[SearchResult] = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            row = self.id_map.iloc[idx]
            if not include_eval_only and _is_eval_only(row):
                continue
            if not _matches(row, filters):
                continue
            results.append(self._row_to_result(int(idx), float(score)))
            if len(results) >= top_k:
                break
        return results

    # ------------------------------------------------------------------ public
    # ---- mobile robustness (Plan Phase 3): test-time augmentation + query expansion ----
    def _tta_views(self, pil: Image.Image) -> list[Image.Image]:
        w, h = pil.size
        cw, ch = int(w * 0.85), int(h * 0.85)
        lx, ty = (w - cw) // 2, (h - ch) // 2
        return [
            pil,                                       # full
            pil.crop((lx, ty, lx + cw, ty + ch)),      # centre crop
            ImageOps.mirror(pil),                      # horizontal flip
            pil.crop((0, 0, cw, ch)),                  # top-left crop
            pil.crop((w - cw, h - ch, w, h)),          # bottom-right crop
        ]

    def _embed_query(self, encoder, query: Image.Image) -> np.ndarray:
        """Embed a query image with the given encoder, applying TTA if enabled."""
        if not self.settings.retrieval.use_tta:
            return encoder.embed_image(query)
        vecs = encoder.embed_images(self._tta_views(query))
        m = vecs.mean(axis=0)
        n = np.linalg.norm(m)
        return (m / n).astype("float32") if n > 0 else vecs[0]

    def _rank_fused(self, fused: np.ndarray, top_k: int, filters: dict | None,
                    include_eval_only: bool = False) -> list[SearchResult]:
        results: list[SearchResult] = []
        for idx in np.argsort(-fused):
            row = self.id_map.iloc[idx]
            if not include_eval_only and _is_eval_only(row):
                continue
            if not _matches(row, filters):
                continue
            results.append(self._row_to_result(int(idx), float(fused[idx])))
            if len(results) >= top_k:
                break
        return results

    def _expand_query(self, qvec: np.ndarray) -> np.ndarray:
        """Average query expansion: blend the query with its top-k neighbour vectors, renormalise."""
        k = min(self.settings.retrieval.rerank_k, self.dino_index.ntotal)
        scores, ids = self.dino_index.search(qvec.reshape(1, -1).astype("float32"), k)
        vecs, weights = [qvec], [1.0]
        for s, i in zip(scores[0], ids[0]):
            if i < 0:
                continue
            vecs.append(self.dino_index.reconstruct(int(i)))
            weights.append(max(0.0, float(s)) * self.settings.retrieval.rerank_alpha)
        m = np.average(np.stack(vecs), axis=0, weights=np.array(weights, dtype="float32"))
        n = np.linalg.norm(m)
        return (m / n).astype("float32") if n > 0 else qvec

    def search_by_image(self, img: Image.Image, top_k: int | None = None,
                        filters: dict | None = None, alpha: float | None = None,
                        include_eval_only: bool = False,
                        prepared: Image.Image | None = None) -> list[SearchResult]:
        """Image search. `alpha` is the DESIGN weight (1.0=pure design, 0.0=pure colour);
        defaults to settings.retrieval.color_alpha. The colour axis (LAB palette + EMD)
        only runs when alpha < 1, so pure-design queries stay fast.

        Pass `prepared` (the output of prepare_query/prepare_query_with_meta) to reuse an
        already-preprocessed image — lets the server segment/rectify once for both the
        before/after preview and the search, instead of running SAM3 twice."""
        top_k = top_k or self.settings.search.default_top_k
        query = prepared if prepared is not None else prepare_query(img, self.settings)

        # Two-axis: blend grayscale-DINO structure similarity with LAB/EMD colour similarity.
        if self._color_cent is not None and self.settings.retrieval.two_axis:
            a = self.settings.retrieval.color_alpha if alpha is None else float(alpha)
            svec = self._embed_query(self.dino, query)
            struct = self._dino_mat @ svec                       # cosine vs every rug, (N,)
            qc, qw = color_profile(query)
            cdist = color_distances(qc, qw, self._color_cent, self._color_wts)
            # Cache the alpha-independent axes so moving the design/colour slider only re-fuses
            # (instant) instead of re-running SAM3 + DINO + colour from scratch (see rerank()).
            self._last_axes = {"struct": struct, "cdist": cdist}
            return self._rank_fused(self._fuse(struct, cdist, a), top_k, filters, include_eval_only)

        if self.marqo is not None and self.settings.retrieval.ensemble:
            a = self.settings.retrieval.ensemble_alpha
            dvec = self._embed_query(self.dino, query)
            mvec = self._embed_query(self.marqo, query)
            fused = a * (self._dino_mat @ dvec) + (1 - a) * (self._marqo_mat @ mvec)
            return self._rank_fused(fused, top_k, filters, include_eval_only)

        vec = self._embed_query(self.dino, query)
        if self.settings.retrieval.use_reranking:
            vec = self._expand_query(vec)
        return self._search_index(self.dino_index, vec, top_k, filters, include_eval_only)

    @staticmethod
    def _fuse(struct, cdist, a):
        """Blend the two axes by the design weight `a` (1.0 = pure design, 0.0 = pure colour)."""
        a = min(1.0, max(0.0, float(a)))
        if a >= 0.999:
            return struct                                        # pure design
        return a * _minmax(struct) + (1 - a) * (1.0 - _minmax(cdist))

    def rerank(self, alpha: float, top_k: int | None = None, filters: dict | None = None,
               include_eval_only: bool = False) -> list[SearchResult]:
        """Re-rank the LAST image query at a new design/colour weight — no SAM3/DINO/colour
        recompute, just a re-fuse of the cached axes. Returns [] if there's no cached query."""
        axes = getattr(self, "_last_axes", None)
        if axes is None:
            return []
        top_k = top_k or self.settings.search.default_top_k
        fused = self._fuse(axes["struct"], axes["cdist"], alpha)
        return self._rank_fused(fused, top_k, filters, include_eval_only)

    def search_by_text(self, text: str, top_k: int | None = None,
                       filters: dict | None = None,
                       include_eval_only: bool = False) -> list[SearchResult]:
        top_k = top_k or self.settings.search.default_top_k
        vec = self.clip.embed_text(text)
        return self._search_index(self.clip_index, vec, top_k, filters, include_eval_only)

    # convenience for the UI: distinct values per filterable column
    def distinct_values(self, column: str) -> list[str]:
        if column not in self.id_map.columns:
            return []
        vals = (
            self.id_map[column]
            .dropna()
            .astype(str)
            .str.strip()
            .replace("", np.nan)
            .dropna()
            .unique()
            .tolist()
        )
        return sorted(vals)
