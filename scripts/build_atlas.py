"""Pack catalogue thumbnails into ONE texture atlas for the Constellation/Atlas map.

A single atlas (one WebGL texture, one draw call via InstancedMesh) is the only way to
draw ~2000 thumbnails at 60fps. Each rug is fit-inside an 80x80 transparent cell (no
distortion); cells are packed into a near-square grid kept under the 4096px WebGL
max-texture floor. Writes web/constellation/atlas.png + atlas.json (id_map ROW ORDER).

Re-run after build_index/build_color_index (row-order keyed to id_map).

    python -m scripts.build_atlas
"""

from __future__ import annotations

import json
import math

import pandas as pd
from PIL import Image

from carpet_search.config import load_settings

CELL = 80


def main() -> None:
    s = load_settings()
    idmap = pd.read_parquet(s.id_map_path).reset_index(drop=True)
    n = len(idmap)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    if cols * CELL > 4096 or rows * CELL > 4096:
        raise SystemExit(f"atlas {cols * CELL}x{rows * CELL} exceeds 4096 WebGL limit; lower CELL")

    atlas = Image.new("RGBA", (cols * CELL, rows * CELL), (0, 0, 0, 0))
    items = []
    miss = 0
    for i, row in enumerate(idmap.itertuples()):
        sku = str(row.sku)
        col, r = i % cols, i // cols
        aspect = 1.0
        try:
            im = Image.open(s.project_root / row.image_path).convert("RGB")
            aspect = im.width / im.height
            im.thumbnail((CELL, CELL))                       # fit inside, preserve aspect
            cell = Image.new("RGBA", (CELL, CELL), (0, 0, 0, 0))
            cell.paste(im, ((CELL - im.width) // 2, (CELL - im.height) // 2))
            atlas.paste(cell, (col * CELL, r * CELL))
        except Exception as exc:
            miss += 1
            if miss <= 5:
                print(f"  ! {sku}: {exc}")
        items.append({"sku": sku, "col": col, "row": r, "aspect": round(aspect, 3)})

    out_dir = s.project_root / "web" / "constellation"
    out_dir.mkdir(parents=True, exist_ok=True)
    atlas.save(out_dir / "atlas.png", optimize=True)
    json.dump({"cell": CELL, "cols": cols, "rows": rows, "count": n, "items": items},
              open(out_dir / "atlas.json", "w"))
    print(f"wrote {out_dir / 'atlas.png'} ({cols * CELL}x{rows * CELL}px, grid {cols}x{rows}) "
          f"+ atlas.json for {n} rugs ({miss} missing)")


if __name__ == "__main__":
    main()
