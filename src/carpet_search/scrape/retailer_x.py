"""Concrete scraper for one retailer (spec section 6.7).

TEMPLATE / TODO: this prototype ingests a local dataset instead of scraping (see
`carpet_search.ingest`). To enable real scraping, fill in the two methods below for
your chosen retailer and run `python -m scripts.scrape`.

Implementation guidance:
- Inspect the site's network tab first; prefer a hidden JSON product API.
- Otherwise parse JSON-LD: selectolax -> `script[type="application/ld+json"]`.
- HTML selectors only as a last resort.
- Normalise raw values into the schema (e.g. "Beige / Ivory" -> primary colour "beige").
- Pick the clean/flat product shot as canonical; save room shots to lifestyle/.
"""

from __future__ import annotations

from typing import Iterator

from ..schema import CarpetRecord
from .base import ScraperBase


class RetailerXScraper(ScraperBase):
    BASE_URL = "https://example.com"  # TODO: set the retailer base URL

    def collect_listing_urls(self) -> Iterator[str]:
        raise NotImplementedError(
            "retailer_x.collect_listing_urls is a template. Fill in category pagination "
            "for your chosen retailer, or use local ingestion: python -m scripts.ingest"
        )
        yield  # pragma: no cover  (marks this as a generator)

    def parse_product(self, html_or_json: str) -> CarpetRecord | None:
        raise NotImplementedError(
            "retailer_x.parse_product is a template. Extract fields via hidden JSON API / "
            "JSON-LD / selectors and return a normalised CarpetRecord."
        )
