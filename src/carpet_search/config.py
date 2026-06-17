"""Load config.yaml into a typed, path-resolved Settings object.

This is the single import point for configuration across the codebase:

    from carpet_search.config import load_settings
    settings = load_settings()           # reads ./config.yaml by default

All relative paths in config.yaml are resolved to absolute paths against the
directory that contains config.yaml (treated as the project root).
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class Paths(BaseModel):
    catalogue_images: Path
    lifestyle_images: Path
    metadata_csv: Path
    index_dir: Path


class FamilySpec(BaseModel):
    pattern: str
    title: str


class DatasetCfg(BaseModel):
    source_dir: Path
    families: dict[str, FamilySpec]
    canonical_suffix: str = ""
    query_suffixes: list[str] = Field(default_factory=lambda: ["g", "l", "t"])
    default_shape: str = "rectangle"


class Models(BaseModel):
    dino_name: str
    dino_dim: int
    clip_name: str
    clip_pretrained: str
    clip_dim: int
    image_size: int = 224
    marqo_name: str = "hf-hub:Marqo/marqo-fashionSigLIP"  # 2nd image encoder for the ensemble
    marqo_dim: int = 768
    sam3_name: str = "facebook/sam3"   # gated HF model for carpet segmentation (needs HF_TOKEN)
    sam3_prompt: str = "carpet"        # text prompt SAM3 uses to locate the rug
    sam3_api_url: str = ""             # hosted SAM3 corner service (POST file -> {corners,...})
    sam3_api_timeout: float = 60.0     # seconds to wait on the corner API


class Preprocess(BaseModel):
    use_segmentation: bool = False
    use_rectification: bool = False
    grayscale: bool = False        # embed DINO image path in grayscale (removes colour bias)
    use_white_balance: bool = False  # gray-world white balance on the query (colour-cast fix)
    use_clahe: bool = False          # CLAHE contrast normalisation on the query
    segmentation_backend: str = "none"   # "none" | "grabcut" | "sam3"; mask source for seg + rectify
    sam3_fallback_grabcut: bool = True   # if SAM3 errors/unauthenticated, degrade to a GrabCut mask


class RetrievalCfg(BaseModel):
    use_tta: bool = False          # test-time augmentation: average multi-crop/flip query embeddings
    use_reranking: bool = False    # average query expansion (blend query with its top neighbours)
    rerank_k: int = 10             # neighbours used for query expansion
    rerank_alpha: float = 0.5      # weight of neighbour vectors relative to the query
    ensemble: bool = False         # fuse DINOv2 + Marqo image scores (needs marqo.faiss)
    ensemble_alpha: float = 0.5    # weight on DINOv2 (1-alpha on Marqo) in the fused score
    two_axis: bool = False         # structure (grayscale DINO) + colour (LAB/EMD) with a user alpha
    color_alpha: float = 0.6       # weight on DESIGN/structure; (1-alpha) on COLOUR


class SearchCfg(BaseModel):
    default_top_k: int = 12
    overfetch_factor: int = 5


class AttributesCfg(BaseModel):
    auto_tag: bool = False        # run CLIP zero-shot tagging during ingest
    palette_size: int = 3
    min_confidence: float = 0.0   # below this, leave a zero-shot attribute blank (0 = always fill)


class EvalCfg(BaseModel):
    ks: list[int] = Field(default_factory=lambda: [1, 5, 10])
    synth_seed: int = 42
    synth_per_sku: int = 1
    synth_profile: str = "default"  # "default" or "mobile" (harder glare/blur/JPEG queries)


class ScrapeCfg(BaseModel):
    retailer: str
    max_items: int
    rate_limit_seconds: float
    user_agent: str


class UICfg(BaseModel):
    # facet columns offered as filter dropdowns (only those with >1 distinct value render).
    # Defaults to the trustworthy attributes; material / has_medallion / has_border are
    # computed and stored but excluded here (CLIP zero-shot material is silk-biased,
    # medallion under-detects, and border is "yes" for ~all rugs — see README).
    facets: list[str] = Field(default_factory=lambda: ["color", "shape", "pattern", "style"])
    show_explanations: bool = True  # show a "why similar" line under image results


class Settings(BaseModel):
    project_root: Path
    paths: Paths
    dataset: DatasetCfg
    models: Models
    preprocess: Preprocess
    search: SearchCfg
    eval: EvalCfg
    scrape: ScrapeCfg
    attributes: AttributesCfg = Field(default_factory=AttributesCfg)
    ui: UICfg = Field(default_factory=UICfg)
    retrieval: RetrievalCfg = Field(default_factory=RetrievalCfg)

    # --- derived index artifact locations -------------------------------------
    @property
    def dino_index_path(self) -> Path:
        return self.paths.index_dir / "dino.faiss"

    @property
    def clip_index_path(self) -> Path:
        return self.paths.index_dir / "clip.faiss"

    @property
    def marqo_index_path(self) -> Path:
        return self.paths.index_dir / "marqo.faiss"

    @property
    def color_index_path(self) -> Path:
        return self.paths.index_dir / "color.npz"

    @property
    def id_map_path(self) -> Path:
        return self.paths.index_dir / "id_map.parquet"


def _resolve(value, root: Path) -> str:
    p = Path(value)
    return str(p if p.is_absolute() else (root / p))


def load_settings(path: str | Path = "config.yaml") -> Settings:
    cfg_path = Path(path).resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"config file not found: {cfg_path}")
    root = cfg_path.parent

    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    # Resolve every path-like value against the project root.
    for key, value in raw["paths"].items():
        raw["paths"][key] = _resolve(value, root)
    raw["dataset"]["source_dir"] = _resolve(raw["dataset"]["source_dir"], root)
    raw["project_root"] = str(root)

    return Settings(**raw)
