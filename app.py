"""Gradio UI (spec section 6.6 + Plan Phase 5).

Two tabs:
  - Image search: upload/drag a carpet photo -> DINOv2 nearest neighbours.
  - Text search:  type a query -> CLIP text->image matches.
Filters are CONFIG-DRIVEN (one dropdown per facet in settings.ui.facets that has >1
distinct value), result tiles show extracted attributes, and image results get a
"why similar" line comparing the query's (cheap, classical) attributes to each match.
Models + index load lazily on the first search so the UI appears quickly.

    python app.py
"""

from __future__ import annotations

import pandas as pd

from carpet_search.config import load_settings

settings = load_settings()

# facet column -> display label
LABELS = {
    "color": "Colour", "shape": "Shape", "material": "Material", "pattern": "Pattern",
    "style": "Style", "has_border": "Border", "has_medallion": "Medallion",
}

_engine = None


def engine():
    """Lazily construct the search engine (loads models + FAISS index once)."""
    global _engine
    if _engine is None:
        from carpet_search.search import CarpetSearch

        _engine = CarpetSearch(settings)
    return _engine


def _choices(column: str) -> list[str]:
    try:
        df = pd.read_parquet(settings.id_map_path)
        vals = (
            df[column].dropna().astype(str).str.strip()
            .replace("", pd.NA).dropna().unique().tolist()
        )
        return ["Any"] + sorted(vals)
    except Exception:
        return ["Any"]


def _active_facets() -> list[str]:
    """Facets from config that have at least one real value in the catalogue."""
    return [c for c in settings.ui.facets if len(_choices(c)) > 1]


def _query_attrs(img):
    """Cheap classical attributes of the query image (no model) for 'why similar'."""
    try:
        from carpet_search.attributes import derive_color, derive_palette, derive_shape_aspect
        from carpet_search.preprocess import to_pil

        q = to_pil(img)
        return {
            "color": derive_color(q),
            "shape": derive_shape_aspect(q)[0],
            "palette": set(derive_palette(q, settings.attributes.palette_size)),
        }
    except Exception:
        return None


def _palette_list(value) -> list[str]:
    return [x.strip() for x in str(value or "").split(",") if x.strip() and x.strip() != "nan"]


def _is_real(v) -> bool:
    return v is not None and str(v).strip() not in ("", "nan", "None")


def _caption(r, qattrs=None) -> str:
    m = r.metadata
    lines = [f"{r.score:.3f} · {m.get('title', r.sku)}"]
    # Show only the trustworthy attributes (material/medallion are stored but unreliable here).
    attrs = [str(m[k]) for k in ("color", "pattern", "style", "shape") if _is_real(m.get(k))]
    if attrs:
        lines.append(" · ".join(attrs))
    if qattrs and settings.ui.show_explanations:
        shared = []
        if _is_real(m.get("color")) and str(m["color"]) == qattrs["color"]:
            shared.append(f"colour {qattrs['color']}")
        if _is_real(m.get("shape")) and str(m["shape"]) == qattrs["shape"]:
            shared.append(f"shape {qattrs['shape']}")
        common = qattrs["palette"] & set(_palette_list(m.get("palette")))
        if common:
            shared.append("palette " + "/".join(sorted(common)))
        if shared:
            lines.append("↔ shares " + ", ".join(shared))
    return "\n".join(lines)


def _gallery(results, qattrs=None) -> list[tuple[str, str]]:
    return [(r.image_path, _caption(r, qattrs)) for r in results]


def _filters(facet_cols, values) -> dict:
    return {col: v for col, v in zip(facet_cols, values) if v and v != "Any"}


def build_ui():
    import gradio as gr

    facet_cols = _active_facets()

    # analytics_enabled=False avoids Gradio's telemetry network call, which can hang on
    # TLS-intercepting corporate networks during launch.
    with gr.Blocks(title="Carpet Visual Search", analytics_enabled=False) as demo:
        gr.Markdown(
            "# 🧶 Carpet Visual Search\n"
            "Image→image similarity (DINOv2) and text→image search (CLIP) over a local "
            "carpet catalogue, with AI-extracted attribute filters. *Many catalogue scans are "
            "grayscale, so colour text-queries are unreliable; pattern/structure queries work best.*"
        )

        facet_dropdowns = {}
        with gr.Row():
            for col in facet_cols:
                facet_dropdowns[col] = gr.Dropdown(
                    _choices(col), value="Any", label=LABELS.get(col, col)
                )
            topk = gr.Slider(1, 48, value=settings.search.default_top_k, step=1, label="Top-K")

        facet_inputs = list(facet_dropdowns.values())

        def search_image(img, top_k, *facet_values):
            if img is None:
                return [], "⚠️ Upload an image to search."
            try:
                results = engine().search_by_image(
                    img, top_k=int(top_k), filters=_filters(facet_cols, facet_values)
                )
            except Exception as exc:
                return [], f"❌ Search failed: {exc}"
            if not results:
                return [], "No matches — try relaxing the filters."
            return _gallery(results, _query_attrs(img)), f"✅ {len(results)} result(s)."

        def search_text(text, top_k, *facet_values):
            if not text or not text.strip():
                return [], "⚠️ Enter a text query."
            try:
                results = engine().search_by_text(
                    text.strip(), top_k=int(top_k), filters=_filters(facet_cols, facet_values)
                )
            except Exception as exc:
                return [], f"❌ Search failed: {exc}"
            if not results:
                return [], "No matches — try relaxing the filters."
            return _gallery(results), f"✅ {len(results)} result(s)."

        with gr.Tabs():
            with gr.Tab("🖼️ Image search"):
                img_in = gr.Image(type="pil", label="Upload / drag a carpet photo", height=320)
                img_btn = gr.Button("Search by image", variant="primary")
                img_status = gr.Markdown()
                img_gallery = gr.Gallery(label="Results", columns=4, height=560, object_fit="cover")
                img_btn.click(search_image, [img_in, topk, *facet_inputs], [img_gallery, img_status])

            with gr.Tab("🔤 Text search"):
                txt_in = gr.Textbox(label="Describe the carpet",
                                    placeholder="e.g. ornate floral medallion persian rug")
                txt_btn = gr.Button("Search by text", variant="primary")
                txt_status = gr.Markdown()
                txt_gallery = gr.Gallery(label="Results", columns=4, height=560, object_fit="cover")
                txt_btn.click(search_text, [txt_in, topk, *facet_inputs], [txt_gallery, txt_status])
                txt_in.submit(search_text, [txt_in, topk, *facet_inputs], [txt_gallery, txt_status])

    return demo


if __name__ == "__main__":
    import gradio as gr

    build_ui().launch(
        theme=gr.themes.Soft(),
        server_name="127.0.0.1",
        show_error=True,
    )
