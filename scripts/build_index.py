"""Phases 1/3: embed the catalogue and build FAISS indices.

    python -m scripts.build_index
"""

from __future__ import annotations

from carpet_search.config import load_settings
from carpet_search.index import build_indices


def main() -> None:
    settings = load_settings()
    build_indices(settings)


if __name__ == "__main__":
    main()
