"""Build the colour-axis index (color.npz) for the two-axis engine.

Run after build_index (it aligns to the existing id_map.parquet row order).

    python -m scripts.build_color_index
"""

from __future__ import annotations

from carpet_search.color_index import build_color_index
from carpet_search.config import load_settings


def main() -> None:
    build_color_index(load_settings())


if __name__ == "__main__":
    main()
