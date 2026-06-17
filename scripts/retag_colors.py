"""Retag the `color` + `palette` fields from the LAB colour profiles (color.npz).

The retailer's own colour tags are unreliably ordered and the old HSV `derive_color`
mis-named many rugs. The KMeans-LAB profiles we built for the colour-search axis are
pixel-grounded, so we name colours from those instead. Pure metadata surgery — updates
id_map.parquet (+ metadata.csv by sku), no re-embedding.

    python -m scripts.retag_colors
"""

from __future__ import annotations

import pandas as pd

from carpet_search.color_index import load_color_index, profile_color_names
from carpet_search.config import load_settings


def main() -> None:
    s = load_settings()
    cents, wts = load_color_index(s)
    idmap = pd.read_parquet(s.id_map_path).reset_index(drop=True)
    if len(idmap) != len(cents):
        raise SystemExit(f"alignment mismatch: id_map {len(idmap)} vs profiles {len(cents)}")

    before = idmap[["sku", "color", "palette"]].head(6).copy()
    colors, palettes = [], []
    for i in range(len(idmap)):
        prim, pal = profile_color_names(cents[i], wts[i])
        colors.append(prim)
        palettes.append(", ".join(pal))
    idmap["color"] = colors
    idmap["palette"] = palettes
    
    idmap.to_parquet(s.id_map_path, index=False)

    # Keep metadata.csv consistent (by sku, so row order is irrelevant).
    meta = pd.read_csv(s.paths.metadata_csv)
    cmap = dict(zip(idmap["sku"], colors))
    pmap = dict(zip(idmap["sku"], palettes))
    meta["color"] = meta["sku"].map(cmap).fillna(meta["color"])
    meta["palette"] = meta["sku"].map(pmap).fillna(meta["palette"])
    meta.to_csv(s.paths.metadata_csv, index=False)

    print(f"retagged {len(idmap)} rows from LAB colour profiles")
    print("BEFORE:\n" + before.to_string(index=False))
    print("AFTER:\n" + idmap[["sku", "color", "palette"]].head(6).to_string(index=False))


if __name__ == "__main__":
    main()
