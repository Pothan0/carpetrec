"""Build the Marqo image index (marqo.faiss) for the DINOv2+Marqo ensemble.

Additive: leaves dino.faiss / clip.faiss / id_map.parquet untouched. Run after build_index.

    python -m scripts.build_marqo_index
"""

from __future__ import annotations

from carpet_search.config import load_settings
from carpet_search.index import build_marqo_index


def main() -> None:
    build_marqo_index(load_settings())


if __name__ == "__main__":
    main()
