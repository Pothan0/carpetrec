"""Phase 0 (retailer framework): run the scraper for the configured retailer.

    python -m scripts.scrape

NOTE: this prototype's working catalogue comes from local ingestion
(`python -m scripts.ingest`). This entrypoint exercises the scraper framework, which
is a template until retailer_x.py selectors are filled in.
"""

from __future__ import annotations

from carpet_search.config import load_settings
from carpet_search.scrape.retailer_x import RetailerXScraper


def main() -> None:
    settings = load_settings()
    scraper = RetailerXScraper(settings)
    scraper.run()


if __name__ == "__main__":
    main()
