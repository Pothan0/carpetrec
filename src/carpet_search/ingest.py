"""Local catalogue ingestion (Phase 0, adapted from scraping to on-disk data).

Reads the `Dataset/<family>/NNN[suffix].<ext>` layout, where each design id has a
base image (canonical/indexed shot) plus g/l/t variants (reserved as eval queries).
Produces:

    data/catalogue/images/{sku}.jpg        # canonical, indexed
    data/catalogue/lifestyle/{sku}__{v}.jpg # variant queries (eval only, never indexed)
    data/catalogue/metadata.csv             # one row per SKU (schema.METADATA_COLUMNS)

Metadata policy for this dataset: `pattern` and `shape` come from the family config,
`color` is derived from the canonical image, and genuinely-unknown fields
(size/material/price/source_url) are left blank rather than fabricated.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from .attributes import derive_color
from .config import Settings, load_settings
from .schema import METADATA_COLUMNS, CarpetRecord

# filename stem -> (design id, variant suffix); e.g. "001g" -> ("001", "g"), "001" -> ("001", "")
_STEM_RE = re.compile(r"^(\d+)([a-zA-Z]*)$")
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


# Colour derivation now lives in attributes.py (single home for feature extraction);
# `derive_color` is imported above.


# --------------------------------------------------------------------------- ingest
def _group_by_design(family_dir: Path) -> dict[str, dict[str, Path]]:
    """{design_id: {suffix: path}} for one family folder."""
    groups: dict[str, dict[str, Path]] = defaultdict(dict)
    for f in sorted(family_dir.iterdir()):
        if not f.is_file() or f.suffix.lower() not in _IMAGE_EXTS:
            continue
        m = _STEM_RE.match(f.stem)
        if not m:
            continue
        design_id, suffix = m.group(1), m.group(2).lower()
        groups[design_id][suffix] = f
    return groups


def _save_jpg(src: Path, dest: Path) -> bool:
    """Re-encode any supported image to RGB JPEG at dest. Returns False on failure."""
    try:
        with Image.open(src) as im:
            im.convert("RGB").save(dest, format="JPEG", quality=95)
        return True
    except Exception as exc:  # corrupt / unreadable image — skip and log
        print(f"  ! failed to read {src.name}: {exc}")
        return False


def ingest(settings: Settings) -> pd.DataFrame:
    ds = settings.dataset
    images_out = settings.paths.catalogue_images
    life_out = settings.paths.lifestyle_images
    images_out.mkdir(parents=True, exist_ok=True)
    life_out.mkdir(parents=True, exist_ok=True)
    root = settings.project_root

    records: list[CarpetRecord] = []
    n_queries = 0

    for family, spec in ds.families.items():
        family_dir = ds.source_dir / family
        if not family_dir.is_dir():
            print(f"! family folder missing, skipping: {family_dir}")
            continue
        groups = _group_by_design(family_dir)
        print(f"[{family}] {len(groups)} designs")

        for design_id, variants in tqdm(sorted(groups.items()), desc=family, unit="sku"):
            canonical = variants.get(ds.canonical_suffix)
            if canonical is None:
                print(f"  ! {family}/{design_id}: no canonical (base) image, skipping")
                continue

            sku = f"{family}_{design_id}"
            dest = images_out / f"{sku}.jpg"
            if not _save_jpg(canonical, dest):
                continue

            with Image.open(dest) as im:
                color = derive_color(im)

            # Variant images -> lifestyle/ as eval queries (never indexed).
            for suffix in ds.query_suffixes:
                vpath = variants.get(suffix)
                if vpath is None:
                    continue
                vdest = life_out / f"{sku}__{suffix}.jpg"
                if _save_jpg(vpath, vdest):
                    n_queries += 1

            rel = dest.relative_to(root).as_posix()
            records.append(
                CarpetRecord(
                    sku=sku,
                    title=f"{spec.title} {design_id}",
                    image_path=rel,
                    color=color,
                    pattern=spec.pattern,
                    shape=ds.default_shape,
                )
            )

    df = pd.DataFrame([r.csv_row() for r in records], columns=METADATA_COLUMNS)
    settings.paths.metadata_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(settings.paths.metadata_csv, index=False)

    print(
        f"\nWrote {len(df)} SKUs -> {settings.paths.metadata_csv}\n"
        f"  catalogue images: {images_out}\n"
        f"  lifestyle (eval) queries: {n_queries} -> {life_out}\n"
        f"  colour distribution: "
        + ", ".join(f"{k}={v}" for k, v in df['color'].value_counts().items())
    )
    return df


if __name__ == "__main__":
    ingest(load_settings())
