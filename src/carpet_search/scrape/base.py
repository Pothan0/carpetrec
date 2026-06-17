"""Scraper framework (spec section 6.7).

NOTE: For this prototype the working data path is local ingestion of the on-disk
`Dataset/` folder (see `carpet_search.ingest`). This module is the framework the spec
asks for so a real retailer can be plugged in later by implementing one subclass
(see `retailer_x.py`). The `run()` loop here is generic and functional; only the two
abstract methods are retailer-specific.
"""

from __future__ import annotations

import time
import urllib.robotparser
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator
from urllib.parse import urlparse

import pandas as pd

from ..config import Settings
from ..schema import METADATA_COLUMNS, CarpetRecord


class ScraperBase(ABC):
    def __init__(self, settings: Settings):
        self.settings = settings
        self.ua = settings.scrape.user_agent
        self._robots: dict[str, urllib.robotparser.RobotFileParser] = {}

    # ----- retailer-specific (implement these in a subclass) ------------------
    @abstractmethod
    def collect_listing_urls(self) -> Iterator[str]:
        """Yield product-detail URLs by paginating category pages."""

    @abstractmethod
    def parse_product(self, html_or_json: str) -> CarpetRecord | None:
        """Parse one product page into a normalised CarpetRecord (or None to skip).

        Extraction priority: (1) hidden JSON product API, (2) JSON-LD
        <script type="application/ld+json">, (3) HTML selectors as last resort.
        """

    # ----- generic plumbing ---------------------------------------------------
    def _allowed(self, url: str) -> bool:
        parsed = urlparse(url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        if base not in self._robots:
            rp = urllib.robotparser.RobotFileParser()
            rp.set_url(f"{base}/robots.txt")
            try:
                rp.read()
            except Exception:
                rp = None  # if robots can't be read, fall through to allow
            self._robots[base] = rp
        rp = self._robots[base]
        return True if rp is None else rp.can_fetch(self.ua, url)

    def _fetch(self, url: str) -> str | None:
        import httpx

        try:
            resp = httpx.get(url, headers={"User-Agent": self.ua}, timeout=20.0, follow_redirects=True)
            resp.raise_for_status()
            return resp.text
        except Exception as exc:
            print(f"  ! fetch failed {url}: {exc}")
            return None

    def _download_image(self, url: str, dest: Path) -> bool:
        import httpx

        try:
            resp = httpx.get(url, headers={"User-Agent": self.ua}, timeout=30.0, follow_redirects=True)
            resp.raise_for_status()
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(resp.content)
            return True
        except Exception as exc:
            print(f"  ! image download failed {url}: {exc}")
            return False

    def run(self, max_items: int | None = None, rate_limit: float | None = None) -> None:
        max_items = max_items or self.settings.scrape.max_items
        rate_limit = self.settings.scrape.rate_limit_seconds if rate_limit is None else rate_limit

        records: list[CarpetRecord] = []
        for url in self.collect_listing_urls():
            if len(records) >= max_items:
                break
            if not self._allowed(url):
                print(f"  - robots.txt disallows {url}, skipping")
                continue
            html = self._fetch(url)
            if html is None:
                continue
            rec = self.parse_product(html)
            if rec is not None:
                records.append(rec)
            time.sleep(rate_limit)  # politeness

        df = pd.DataFrame([r.csv_row() for r in records], columns=METADATA_COLUMNS)
        self.settings.paths.metadata_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.settings.paths.metadata_csv, index=False)
        print(f"Scraped {len(df)} products -> {self.settings.paths.metadata_csv}")
