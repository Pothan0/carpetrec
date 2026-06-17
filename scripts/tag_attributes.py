"""Phase 2: AI attribute tagging over an existing catalogue (no re-ingest, no re-embed).

Reuses the CLIP image vectors already stored in clip.faiss (reconstructed), runs CLIP
zero-shot tagging for pattern/material/style/border/medallion, and classical extraction
for palette/shape/aspect. Fills only BLANK fields (never overwrites known-good values),
rewrites metadata.csv, and patches id_map.parquet in place so the UI gets facets without
rebuilding the index.

    python -m scripts.tag_attributes
"""

from __future__ import annotations

import faiss
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from carpet_search.attributes import AttributeTagger, derive_palette, derive_shape_aspect
from carpet_search.config import load_settings
from carpet_search.embedding import get_clip
from carpet_search.index import ID_MAP_COLUMNS
from carpet_search.schema import METADATA_COLUMNS


def _blank(v) -> bool:
    return v is None or (isinstance(v, float) and np.isnan(v)) or str(v).strip() == ""


def main() -> None:
    s = load_settings()

    df = pd.read_csv(s.paths.metadata_csv)
    for c in METADATA_COLUMNS:
        if c not in df.columns:
            df[c] = None

    # Reconstruct CLIP image vectors (already L2-normalised) keyed by sku.
    clip_index = faiss.read_index(str(s.clip_index_path))
    idmap = pd.read_parquet(s.id_map_path)
    vecs = clip_index.reconstruct_n(0, clip_index.ntotal)
    sku2vec = {idmap.iloc[i]["sku"]: vecs[i] for i in range(len(idmap))}

    print(f"Tagging {len(df)} SKUs (CLIP zero-shot + classical); loading CLIP...")
    tagger = AttributeTagger(
        get_clip(s.models.clip_name, s.models.clip_pretrained),
        min_confidence=s.attributes.min_confidence,
    )

    cols = ["material", "pattern", "shape", "palette", "aspect",
            "has_border", "has_medallion", "style"]
    new = {c: [] for c in cols}
    clip_pattern_pred, existing_pattern = [], []  # for a zero-shot accuracy proxy

    for _, row in tqdm(df.iterrows(), total=len(df), desc="tag", unit="sku"):
        try:
            img = Image.open(s.project_root / row["image_path"]).convert("RGB")
        except Exception:
            for c in cols:
                new[c].append(row.get(c))
            continue

        palette = ", ".join(derive_palette(img, s.attributes.palette_size))
        shape_guess, aspect = derive_shape_aspect(img)
        vec = sku2vec.get(row["sku"])
        tags = tagger.tag_from_vec(vec) if vec is not None else {}

        # Always-new columns
        new["palette"].append(palette)
        new["aspect"].append(aspect)
        new["has_border"].append(tags.get("has_border", ""))
        new["has_medallion"].append(tags.get("has_medallion", ""))
        new["style"].append(tags.get("style", ""))
        # Fill-if-blank columns (never overwrite known-good ground truth)
        new["material"].append(row["material"] if not _blank(row.get("material")) else tags.get("material", ""))
        new["pattern"].append(row["pattern"] if not _blank(row.get("pattern")) else tags.get("pattern", ""))
        new["shape"].append(row["shape"] if not _blank(row.get("shape")) else shape_guess)

        if tags.get("pattern"):
            clip_pattern_pred.append(tags["pattern"])
            existing_pattern.append(str(row.get("pattern", "")).strip().lower())

    for c, vals in new.items():
        df[c] = vals
    df = df[METADATA_COLUMNS]
    df.to_csv(s.paths.metadata_csv, index=False)

    # Patch id_map.parquet in place (no reindex needed).
    m = df.set_index("sku")
    for c in ID_MAP_COLUMNS:
        if c in ("sku", "image_path", "title"):
            continue
        if c in m.columns:
            idmap[c] = idmap["sku"].map(m[c])
    for c in ID_MAP_COLUMNS:
        if c not in idmap.columns:
            idmap[c] = ""
    idmap = idmap[ID_MAP_COLUMNS]
    idmap.to_parquet(s.id_map_path, index=False)

    # --- report ---
    def dist(col):
        return ", ".join(f"{k}={v}" for k, v in df[col].value_counts().head(8).items())

    print(f"\nWrote {len(df)} rows -> {s.paths.metadata_csv}; patched {s.id_map_path}")
    for col in ["material", "style", "has_border", "has_medallion", "palette"]:
        print(f"  {col:14s}: {dist(col)}")
    if existing_pattern:
        agree = np.mean([p == e for p, e in zip(clip_pattern_pred, existing_pattern)
                         if e in ("floral", "medallion")])
        print(f"\n  CLIP zero-shot pattern vs folder-derived pattern: agreement={agree:.2f} "
              f"(sanity proxy on {sum(e in ('floral','medallion') for e in existing_pattern)} known rugs)")


if __name__ == "__main__":
    main()
