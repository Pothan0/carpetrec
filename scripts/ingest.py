"""Phase 0 (working data path): build the catalogue from the local Dataset/ folder.

    python -m scripts.ingest
"""

from __future__ import annotations

from carpet_search.config import load_settings
from carpet_search.ingest import ingest


def main() -> None:
    settings = load_settings()
    ingest(settings)


if __name__ == "__main__":
    main()
