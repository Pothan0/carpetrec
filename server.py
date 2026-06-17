"""FastAPI backend for the carpet visual-search web UI.

Wraps the existing CarpetSearch engine (unchanged) behind a small JSON API and serves
a custom single-page front-end from web/. The Gradio app (app.py) remains available.

    python server.py            # http://127.0.0.1:8000
    # or: uvicorn server:app --port 8000
"""

from __future__ import annotations

import base64
import io
import json

import pandas as pd
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from carpet_search.config import load_settings

settings = load_settings()
WEB = settings.project_root / "web"

app = FastAPI(title="Tapis — Carpet Visual Search")


@app.middleware("http")
async def _no_cache(request, call_next):
    # Local dev prototype: never let the browser cache assets, so edits always show up.
    resp = await call_next(request)
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return resp


app.mount("/static", StaticFiles(directory=str(WEB)), name="static")
app.mount("/catalogue", StaticFiles(directory=str(settings.paths.catalogue_images)), name="catalogue")

FACET_COLS = ["color", "shape", "pattern", "style"]

_engine = None


def engine():
    """Lazily build the search engine (loads DINOv2 + CLIP + FAISS once)."""
    global _engine
    if _engine is None:
        from carpet_search.search import CarpetSearch

        _engine = CarpetSearch(settings)
    return _engine


@app.on_event("startup")
def _prewarm_engine():
    """Warm the models in a background thread at startup so the first search isn't a 1-2 min
    cold-start. The server still serves static pages immediately while this loads. When SAM3 is
    the segmentation backend, warm it too (guarded: a missing token just logs and serves)."""
    import threading

    def _warm():
        engine()
        if settings.preprocess.segmentation_backend == "sam3":
            try:
                from carpet_search.segmentation import get_sam3

                get_sam3(settings.models.sam3_name)
                print("SAM3 segmenter warmed.")
            except Exception as e:
                print(f"  ! SAM3 prewarm skipped ({type(e).__name__}: {e}); "
                      "image search will fall back to GrabCut/passthrough.")

    threading.Thread(target=_warm, daemon=True).start()


def _data_url(img: Image.Image, max_side: int = 640) -> str:
    """Encode a PIL image as a lossless PNG data URL (Phase 6) for the before/after UI."""
    im = img.convert("RGB")
    im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _idmap(visible_only: bool = False) -> pd.DataFrame:
    df = pd.read_parquet(settings.id_map_path)
    if visible_only and "eval_only" in df.columns:
        df = df[df["eval_only"].astype(str).str.lower() != "true"].reset_index(drop=True)
    return df


def _clean(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).strip()
    return None if s in ("", "nan", "None") else s


def _img_url(sku: str) -> str:
    return f"/catalogue/{sku}.jpg"


def _row_item(row) -> dict:
    return {
        "sku": str(row["sku"]),
        "image_url": _img_url(str(row["sku"])),
        "title": _clean(row.get("title")) or str(row["sku"]),
        "color": _clean(row.get("color")),
        "pattern": _clean(row.get("pattern")),
        "style": _clean(row.get("style")),
        "shape": _clean(row.get("shape")),
        "palette": _clean(row.get("palette")),
    }


def _result_item(r, why=None) -> dict:
    m = r.metadata
    score = max(0.0, min(1.0, float(r.score)))
    return {
        "sku": r.sku,
        "score": round(float(r.score), 4),
        "match": int(round(score * 100)),
        "image_url": _img_url(r.sku),
        "title": _clean(m.get("title")) or r.sku,
        "color": _clean(m.get("color")),
        "pattern": _clean(m.get("pattern")),
        "style": _clean(m.get("style")),
        "shape": _clean(m.get("shape")),
        "palette": _clean(m.get("palette")),
        "why": why,
    }


def _query_attrs(img: Image.Image):
    try:
        from carpet_search.attributes import derive_color, derive_palette, derive_shape_aspect

        return {
            "color": derive_color(img),
            "shape": derive_shape_aspect(img)[0],
            "palette": set(derive_palette(img, settings.attributes.palette_size)),
        }
    except Exception:
        return None


def _why(qa, r) -> str | None:
    if not qa:
        return None
    m = r.metadata
    shared = []
    if _clean(m.get("color")) and str(m["color"]) == qa["color"]:
        shared.append(f"colour {qa['color']}")
    if _clean(m.get("shape")) and str(m["shape"]) == qa["shape"]:
        shared.append(f"shape {qa['shape']}")
    rp = {x.strip() for x in str(m.get("palette") or "").split(",") if x.strip() and x.strip() != "nan"}
    common = qa["palette"] & rp
    if common:
        shared.append("palette " + " / ".join(sorted(common)))
    return ("Shares " + ", ".join(shared)) if shared else None


def _filters(d: dict | None) -> dict:
    return {k: v for k, v in (d or {}).items() if v and v != "Any"}


# ------------------------------------------------------------------ routes
@app.get("/")
def index():
    return FileResponse(str(WEB / "index.html"))


@app.get("/api/facets")
def facets():
    df = _idmap(visible_only=True)
    out = {}
    for c in FACET_COLS:
        if c in df.columns:
            vals = (
                df[c].dropna().astype(str).str.strip()
                .replace("", pd.NA).dropna().unique().tolist()
            )
            out[c] = sorted(vals)
    return out


@app.get("/api/featured")
def featured(n: int = 12):
    df = _idmap(visible_only=True)
    if df.empty:
        return {"items": []}
    step = max(1, len(df) // n)
    idx = list(range(0, len(df), step))[:n]
    return {"items": [_row_item(df.iloc[i]) for i in idx]}


class TextQuery(BaseModel):
    query: str = ""
    top_k: int | None = None
    filters: dict | None = None


@app.post("/api/search/text")
def search_text(q: TextQuery):
    text = (q.query or "").strip()
    if not text:
        return JSONResponse({"results": [], "message": "Enter a search phrase."})
    top_k = int(q.top_k or settings.search.default_top_k)
    res = engine().search_by_text(text, top_k=top_k, filters=_filters(q.filters))
    return {"results": [_result_item(r) for r in res]}


@app.post("/api/search/image")
async def search_image(file: UploadFile = File(...), top_k: int = Form(None),
                       filters: str = Form(None), alpha: float = Form(None)):
    raw = await file.read()
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        return JSONResponse({"results": [], "message": "Could not read that image."}, status_code=400)
    tk = int(top_k) if top_k else settings.search.default_top_k
    try:
        flt = _filters(json.loads(filters) if filters else {})
    except Exception:
        flt = {}
    a = float(alpha) if alpha is not None else None   # design weight; None -> config default

    # Preprocess ONCE (segment/rectify) here, then hand the prepared image to the engine so
    # SAM3 runs a single time and we can show the user the before/after.
    from carpet_search.preprocess import prepare_query_with_meta

    prepared, meta = prepare_query_with_meta(img, settings)
    res = engine().search_by_image(img, top_k=tk, filters=flt, alpha=a, prepared=prepared)
    qa = _query_attrs(prepared)
    return {
        "results": [_result_item(r, _why(qa, r)) for r in res],
        "query_preview": _data_url(img),
        "rectified_preview": _data_url(prepared) if meta.get("rectified") else None,
        "meta": meta,
    }


class RerankQuery(BaseModel):
    alpha: float | None = None
    top_k: int | None = None
    filters: dict | None = None


@app.post("/api/rerank")
def rerank(q: RerankQuery):
    """Re-rank the LAST image query at a new design/colour weight — instant (no SAM3/DINO redo).
    Returns {"results": []} if there's no cached image query (e.g. after a text search)."""
    if q.alpha is None:
        return {"results": []}
    tk = int(q.top_k or settings.search.default_top_k)
    res = engine().rerank(float(q.alpha), top_k=tk, filters=_filters(q.filters))
    return {"results": [_result_item(r) for r in res]}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
