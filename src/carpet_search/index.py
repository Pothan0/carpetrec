"""Build & load FAISS indices + the id_map (spec section 6.4 / 5.3).

Two separate exact (IndexFlatIP) indices over L2-normalised vectors:
  - dino.faiss (768-d) for image->image
  - clip.faiss (512-d) for text->image
id_map.parquet aligns FAISS row order -> sku and carries a denormalised copy of the
metadata columns so search results need no extra join.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from .config import Settings
from .embedding import ClipEncoder, DinoEncoder

ID_MAP_COLUMNS = [
    "sku", "image_path", "title",
    "color", "size", "material", "pattern", "shape", "price", "source_url",
    "palette", "aspect", "has_border", "has_medallion", "style", "eval_only",
]


def _embed_in_batches(encoder, images: list[Image.Image], batch_size: int, desc: str) -> np.ndarray:
    chunks = []
    for i in tqdm(range(0, len(images), batch_size), desc=desc, unit="batch"):
        chunks.append(encoder.embed_images(images[i : i + batch_size]))
    return np.vstack(chunks).astype("float32")


def _embed_paths_in_batches(encoder, paths: list, batch_size: int, desc: str) -> np.ndarray:
    """Stream batches from disk so we never hold the whole catalogue in RAM at once.

    Retailer product shots are large (~10 MB decoded each, occasionally 50 MB+), so
    eagerly loading ~2k of them would need ~20 GB and OOM-kill the process. Loading one
    batch at a time keeps peak memory at batch_size images."""
    chunks = []
    for i in tqdm(range(0, len(paths), batch_size), desc=desc, unit="batch"):
        imgs = []
        for p in paths[i : i + batch_size]:
            with Image.open(p) as im:
                imgs.append(im.convert("RGB"))
        chunks.append(encoder.embed_images(imgs))
    return np.vstack(chunks).astype("float32")


def build_indices(settings: Settings, batch_size: int = 16) -> None:
    import faiss

    df = pd.read_csv(settings.paths.metadata_csv)
    if df.empty:
        raise RuntimeError(f"metadata.csv is empty: {settings.paths.metadata_csv}")

    # Validate readable images WITHOUT holding pixels in RAM (the catalogue is many GB
    # decoded). We keep only paths here; _embed_paths_in_batches loads one batch at a time.
    paths: list = []
    rows: list[dict] = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="checking images", unit="img"):
        path = settings.project_root / row["image_path"]
        try:
            with Image.open(path) as im:
                im.verify()                       # cheap integrity check, no full decode kept
            paths.append(path)
            rows.append(row.to_dict())
        except Exception as exc:
            print(f"  ! skipping {row['sku']} ({path}): {exc}")

    if not paths:
        raise RuntimeError("no images could be loaded; aborting index build")
    print(f"Embedding {len(paths)} catalogue images with DINOv2 + CLIP (streamed)...")

    dino = DinoEncoder(settings.models.dino_name, settings.models.image_size,
                       grayscale=settings.preprocess.grayscale)
    clip = ClipEncoder(settings.models.clip_name, settings.models.clip_pretrained)

    dino_vecs = _embed_paths_in_batches(dino, paths, batch_size, "DINO embed")
    clip_vecs = _embed_paths_in_batches(clip, paths, batch_size, "CLIP embed")

    settings.paths.index_dir.mkdir(parents=True, exist_ok=True)

    dino_index = faiss.IndexFlatIP(settings.models.dino_dim)
    dino_index.add(dino_vecs)
    faiss.write_index(dino_index, str(settings.dino_index_path))

    clip_index = faiss.IndexFlatIP(settings.models.clip_dim)
    clip_index.add(clip_vecs)
    faiss.write_index(clip_index, str(settings.clip_index_path))

    id_map = pd.DataFrame(rows)
    for col in ID_MAP_COLUMNS:
        if col not in id_map.columns:
            id_map[col] = ""
    id_map = id_map[ID_MAP_COLUMNS].reset_index(drop=True)
    id_map.to_parquet(settings.id_map_path, index=False)

    print(
        f"Built indices: {dino_index.ntotal} vectors\n"
        f"  {settings.dino_index_path}  (dim {settings.models.dino_dim})\n"
        f"  {settings.clip_index_path}  (dim {settings.models.clip_dim})\n"
        f"  {settings.id_map_path}"
    )


def build_marqo_index(settings: Settings, batch_size: int = 16) -> None:
    """Embed the catalogue with the Marqo encoder and write marqo.faiss, aligned to the
    EXISTING id_map row order (so ensemble fusion indices line up with dino.faiss)."""
    import faiss

    from .embedding import ClipEncoder

    idmap = pd.read_parquet(settings.id_map_path)
    enc = ClipEncoder(settings.models.marqo_name, None)
    images = [Image.open(settings.project_root / row["image_path"]).convert("RGB")
              for _, row in idmap.iterrows()]
    print(f"Embedding {len(images)} catalogue images with Marqo ({settings.models.marqo_name})...")
    vecs = _embed_in_batches(enc, images, batch_size, "Marqo embed")
    index = faiss.IndexFlatIP(settings.models.marqo_dim)
    index.add(vecs)
    faiss.write_index(index, str(settings.marqo_index_path))
    print(f"Built Marqo index: {index.ntotal} vectors -> {settings.marqo_index_path}")


def load_indices(settings: Settings):
    import faiss

    if not settings.dino_index_path.exists():
        raise FileNotFoundError(
            f"index not found at {settings.dino_index_path}. Run: python -m scripts.build_index"
        )
    dino_index = faiss.read_index(str(settings.dino_index_path))
    clip_index = faiss.read_index(str(settings.clip_index_path))
    id_map = pd.read_parquet(settings.id_map_path)
    return dino_index, clip_index, id_map
