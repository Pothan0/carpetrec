"""Append a Revival Rugs scrape into the shared catalogue (Plan Phase 1a).

    python -m scripts.ingest_revival --dry-run               # validate, write nothing
    python -m scripts.ingest_revival                          # ingest from incoming/output
    python -m scripts.ingest_revival --source some/other/dir  # custom source
"""

from __future__ import annotations

import argparse
from pathlib import Path

from carpet_search.config import load_settings
from carpet_search.ingest_revival import ingest_revival


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="incoming/output",
                    help="folder containing the metadata CSV + images/ subfolder")
    ap.add_argument("--dry-run", action="store_true", help="validate mapping, write nothing")
    ap.add_argument("--limit", type=int, default=None, help="cap number of products (debug)")
    args = ap.parse_args()
    ingest_revival(load_settings(), Path(args.source), dry_run=args.dry_run, limit=args.limit)


if __name__ == "__main__":
    main()
