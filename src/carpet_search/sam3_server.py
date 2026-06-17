from __future__ import annotations

import io
import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image

from carpet_search.segmentation import get_sam3
from carpet_search.rectify import mask_to_quad

app = FastAPI(title="SAM3 wrapper")


@app.post("/get_corners")
async def get_corners(file: UploadFile = File(...)):
    data = await file.read()
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image")

    model_id = os.environ.get("SAM3_MODEL", "facebook/sam3")
    prompt = os.environ.get("SAM3_PROMPT", "carpet")
    try:
        seg = get_sam3(model_id)
        mask = seg.mask(img, prompt=prompt)
    except Exception as e:
        return JSONResponse({"error": f"SAM3 inference failed: {e}"}, status_code=500)

    if mask is None:
        return {"corners": None, "labels": []}

    quad = mask_to_quad(mask)
    if quad is None:
        return {"corners": None, "labels": []}

    return {"corners": quad.tolist(), "labels": []}
