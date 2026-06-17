"""Ingest a Revival Rugs retailer export into the shared catalogue (Plan Phase 1a).

Sibling of `ingest.py`. Where `ingest.py` reads the local Persian `Dataset/` (with g/l/t
eval variants), this reads a retailer scrape: a metadata CSV + a folder of one primary
image per product. It APPENDS modern/mass-market rugs to the same `metadata.csv` so the
catalogue becomes segment-agnostic (Persian + modern in one index).

Key differences from the Persian ingest:
  * Real metadata is used where the retailer provides it — material, size, price, url,
    colours — instead of being left blank or guessed.
  * `shape` is derived from the real `dimensions` string (the catalogue thumbnails are
    square product shots, so image aspect ratio would be a wrong shape signal).
  * No g/l/t variants exist, so nothing is written to lifestyle/ (these rugs add gallery
    diversity + real facets, not same-rug training/eval pairs).

    python -m scripts.ingest_revival --dry-run      # validate mapping, write nothing
    python -m scripts.ingest_revival                # append to metadata.csv + copy images
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

from .attributes import derive_color, derive_palette
from .config import Settings, load_settings
from .ingest import _save_jpg
from .schema import METADATA_COLUMNS, CarpetRecord

SKU_PREFIX = "revival_"
# product_type values that are not actual rugs (drop them).
_NON_RUG_TYPES = {"swatch", "package_protection"}

# one side of a dimensions string, e.g. "6'9\"" -> feet 6, inches 9 ; "8'" -> feet 8
_DIM_SIDE = re.compile(r"(\d+)\s*'\s*(?:(\d+)\s*\")?")


def _dim_inches(side: str) -> float | None:
    m = _DIM_SIDE.search(side or "")
    if not m:
        return None
    feet = int(m.group(1))
    inches = int(m.group(2)) if m.group(2) else 0
    return feet * 12 + inches


def shape_from_dimensions(dim: str) -> str:
    """Derive rug shape from the real dimensions string (not the square thumbnail)."""
    if not isinstance(dim, str) or not dim.strip():
        return ""
    low = dim.lower()
    if "round" in low or "diam" in low or "circle" in low:
        return "round"
    parts = re.split(r"[x×]", dim)
    if len(parts) != 2:
        return ""
    a, b = _dim_inches(parts[0]), _dim_inches(parts[1])
    if not a or not b:
        return ""
    ratio = max(a, b) / min(a, b)
    if ratio < 1.2:
        return "square"
    if ratio >= 2.0:
        return "runner"
    return "rectangle"


def _copy_or_reencode(src: Path, dest: Path) -> bool:
    """Fast path: the retailer images are already JPEG, so copy bytes (no re-encode).
    Fall back to PIL re-encode only for non-JPEG sources."""
    if src.suffix.lower() in (".jpg", ".jpeg"):
        try:
            shutil.copyfile(src, dest)
            return True
        except Exception as exc:
            print(f"  ! copy failed for {src.name} ({exc}); re-encoding")
    return _save_jpg(src, dest)


def _primary(pipe_field: str) -> str:
    """First token of a 'wool|cotton' style field, lowercased; '' if missing."""
    if not isinstance(pipe_field, str) or not pipe_field.strip():
        return ""
    return pipe_field.split("|")[0].strip().lower()


def _palette_from_colors(colors: str) -> str:
    """Real retailer colours -> comma-joined palette string (richer than derived)."""
    if not isinstance(colors, str) or not colors.strip():
        return ""
    return ", ".join(c.strip() for c in colors.split("|") if c.strip())


def _resolve_image(row: pd.Series, images_dir: Path) -> Path | None:
    """Find the on-disk primary image for a product row."""
    for cand in (row.get("local_primary_image"), row.get("handle")):
        if not isinstance(cand, str) or not cand.strip():
            continue
        name = Path(cand).name
        if not name.lower().endswith((".jpg", ".jpeg", ".png")):
            name += ".jpg"
        p = images_dir / name
        if p.exists():
            return p
    return None


def ingest_revival(
    settings: Settings,
    source_dir: Path,
    dry_run: bool = False,
    limit: int | None = None,
) -> pd.DataFrame:
    source_dir = Path(source_dir)
    images_dir = source_dir / "images"
    csvs = sorted(source_dir.glob("*.csv"))
    if not csvs:
        raise SystemExit(f"no metadata CSV found in {source_dir}")
    df = pd.read_csv(csvs[0])
    print(f"loaded {len(df)} rows from {csvs[0].name}")

    # Drop non-rugs, then keep one row per product (handle) — the first/primary variant.
    df = df[~df["product_type"].astype(str).str.lower().isin(_NON_RUG_TYPES)]
    df = df.drop_duplicates(subset="handle", keep="first")
    print(f"{len(df)} unique products after dropping swatches/non-rugs")

    images_out = settings.paths.catalogue_images
    root = settings.project_root
    if not dry_run:
        images_out.mkdir(parents=True, exist_ok=True)

    records: list[CarpetRecord] = []
    preview: list[dict] = []
    skipped_no_image = 0
    rows = list(df.iterrows())
    if limit:
        rows = rows[:limit]

    for _, row in tqdm(rows, desc="revival", unit="rug"):
        handle = str(row.get("handle", "")).strip()
        if not handle:
            continue
        src = _resolve_image(row, images_dir)
        if src is None:
            skipped_no_image += 1
            continue

        sku = SKU_PREFIX + handle
        dest = images_out / f"{sku}.jpg"
        if not dry_run and not _copy_or_reencode(src, dest):
            continue

        # Colour + palette: derive colour from the image (uniform with the Persian set),
        # but prefer the retailer's real colour words for the palette.
        with Image.open(src) as im:
            color = derive_color(im)
            palette_real = _palette_from_colors(row.get("colors"))
            palette = palette_real or ", ".join(derive_palette(im, settings.attributes.palette_size))

        price = row.get("price")
        rec = CarpetRecord(
            sku=sku,
            title=str(row.get("title") or handle),
            image_path=dest.relative_to(root).as_posix(),
            color=color,
            size=str(row.get("dimensions") or row.get("size") or "").strip(),
            material=_primary(row.get("materials")),
            pattern="",                                   # not in feed; tag later if wanted
            shape=shape_from_dimensions(row.get("dimensions")) or "rectangle",
            price=float(price) if pd.notna(price) else None,
            source_url=str(row.get("url") or "").strip(),
            palette=palette,
            style="modern",                               # segment label (Revival = modern brand)
        )
        records.append(rec)
        if len(preview) < 8:
            preview.append({"sku": sku, "title": rec.title[:24], "color": rec.color,
                            "material": rec.material, "shape": rec.shape,
                            "size": rec.size, "price": rec.price})

    new_df = pd.DataFrame([r.csv_row() for r in records], columns=METADATA_COLUMNS)
    print(f"\nbuilt {len(new_df)} records ({skipped_no_image} skipped: no image on disk)")
    print("shape distribution:", new_df["shape"].value_counts().to_dict())
    print("material distribution (top 8):", new_df["material"].value_counts().head(8).to_dict())
    print("\nsample:")
    print(pd.DataFrame(preview).to_string(index=False))

    if dry_run:
        print("\n[dry-run] nothing written. Re-run without --dry-run to ingest.")
        return new_df

    # Combine with the existing catalogue, de-duplicating by sku (idempotent re-runs).
    # Revival is the VISIBLE catalogue; any pre-existing (Persian) rows become the HIDDEN
    # eval gallery — kept in the index as realistic distractors + gold answers for the
    # variant->base eval, but filtered out of every user-facing result (eval_only="true").
    csv_path = settings.paths.metadata_csv
    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        for col in METADATA_COLUMNS:
            if col not in existing.columns:
                existing[col] = ""
        existing = existing[METADATA_COLUMNS]
        hidden_mask = ~existing["sku"].astype(str).str.startswith(SKU_PREFIX)
        existing.loc[hidden_mask, "eval_only"] = "true"
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset="sku", keep="last").reset_index(drop=True)
        print(f"  hid {int(hidden_mask.sum())} pre-existing (Persian) rows as eval_only")
    else:
        combined = new_df
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(csv_path, index=False)
    n_visible = int((combined["eval_only"].astype(str) != "true").sum())
    print(f"\nwrote {len(combined)} total SKUs -> {csv_path} "
          f"({n_visible} visible Revival rugs + {len(combined) - n_visible} hidden eval rugs)")
    print("Next: python -m scripts.build_index  &&  python -m scripts.build_marqo_index")
    return combined


if __name__ == "__main__":
    ingest_revival(load_settings(), Path("incoming/output"))
