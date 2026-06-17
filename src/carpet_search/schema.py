"""Catalogue record + search result schemas (data contracts, spec section 5.2 / 5.4)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from pydantic import BaseModel

# Canonical column order for metadata.csv (kept stable so the CSV is human-diffable).
# The trailing block (palette..style) is added by the attribute-extraction pipeline
# (Plan Phase 2); all optional with empty defaults so older CSVs still load.
METADATA_COLUMNS = [
    "sku",
    "title",
    "image_path",
    "color",
    "size",
    "material",
    "pattern",
    "shape",
    "price",
    "source_url",
    "palette",
    "aspect",
    "has_border",
    "has_medallion",
    "style",
    "eval_only",
]


class CarpetRecord(BaseModel):
    """One catalogue SKU. Mirrors metadata.csv (spec 5.2).

    Fields that are genuinely unknown for the local grayscale/colour scan dataset
    (size, material, price, source_url) default to empty/None rather than being
    fabricated.
    """

    sku: str                      # unique id; also the canonical image filename stem
    title: str                    # product title
    image_path: str               # relative path to the canonical (indexed) image
    color: str = ""               # normalised dominant colour, derived from the image
    size: str = ""                # e.g. "5x8" (feet) — unknown for this dataset
    material: str = ""            # wool/silk/cotton/... — unknown for this dataset
    pattern: str = ""             # geometric/floral/medallion/solid/abstract
    shape: str = "rectangle"      # rectangle/round/runner/square
    price: Optional[float] = None
    source_url: str = ""          # provenance
    # --- attribute-extraction pipeline (Phase 2); empty/None until tagged ---
    palette: str = ""             # comma-joined top dominant colour names
    aspect: Optional[float] = None  # width/height ratio
    has_border: str = ""          # yes/no
    has_medallion: str = ""       # yes/no
    style: str = ""               # persian/oriental/modern/bohemian/...
    eval_only: str = ""           # "true" = kept in the index for eval, hidden from the storefront

    def csv_row(self) -> dict:
        return {c: getattr(self, c) for c in METADATA_COLUMNS}


@dataclass
class SearchResult:
    sku: str
    score: float                  # cosine similarity, 0..1
    image_path: str
    metadata: dict = field(default_factory=dict)  # color, size, material, pattern, shape, price, title
