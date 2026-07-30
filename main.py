"""
Sentinel AI Backend — FastAPI entry point.

Run:
    python main.py
    -or-
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Open http://127.0.0.1:8000/docs for interactive API docs.
"""
import threading
from pathlib import Path

try:                                    # load camera RTSP URLs etc. from .env if present
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

import db
from routers import anpr, asset, biometric, material, safety, vehicles, streams

app = FastAPI(
    title="Sentinel AI Backend",
    version="2.4.1",
    description="Video-analytics backend for the Technomak Sentinel console.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _warmup_ocr():
    from services.anpr_service import _get_paddle, _get_easyocr, read_plate_crop
    import numpy as np
    import cv2
    print("[Startup] Loading OCR models in background…")
    reader = _get_paddle()
    if reader is None or reader == "FAILED":
        _get_easyocr()
    # Loading the model is NOT enough — PaddleOCR compiles its MKL-DNN compute
    # kernels on the FIRST inference of each input SHAPE (~4 s), which otherwise
    # lands on the user's first upload. Run throwaway inferences here so that cost
    # is paid at startup. We warm two representative shapes: a small tight crop and
    # a full-width plate strip (what the contour finder produces on HD frames), so
    # the common scan shapes are already compiled when the first upload arrives.
    try:
        from services.anpr_service import extract_plates_from_frame
        small = np.full((60, 200, 3), 235, np.uint8)
        cv2.putText(small, "AB12CD", (12, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (20, 20, 20), 3)
        read_plate_crop(small)
        strip = np.full((376, 1280, 3), 235, np.uint8)
        cv2.putText(strip, "AB12CD3456", (60, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 5.0, (20, 20, 20), 8)
        read_plate_crop(strip)
        # Full scan path (contour → region crop → OCR) at the 1280-wide work size.
        frame = np.full((720, 1280, 3), 90, np.uint8)
        cv2.rectangle(frame, (520, 470), (760, 545), (235, 235, 235), -1)
        cv2.putText(frame, "AB12CD3456", (532, 527),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (15, 15, 15), 3)
        extract_plates_from_frame(frame)
    except Exception as exc:
        print(f"[Startup] OCR warm-up inference skipped: {exc}")
    print("[Startup] OCR ready — first scan will be fast.")


@app.on_event("startup")
async def on_startup():
    for d in ("data/faces",):   # uploads go to system temp dir (outside OneDrive)
        Path(d).mkdir(parents=True, exist_ok=True)
    db.init_db()
    threading.Thread(target=_warmup_ocr, daemon=True).start()
    print("\n  Sentinel AI backend ready — http://127.0.0.1:8000\n")


@app.get("/", tags=["Health"])
def health_check():
    return {"status": "ok", "version": "2.4.1", "service": "Sentinel AI"}


@app.get("/console", include_in_schema=False)
def console():
    """Serve the operator console over http://localhost so browser features that
    require a secure context (e.g. webcam / getUserMedia) work reliably."""
    return FileResponse("technomak-video-analytics-console.html")


app.include_router(streams.router,   tags=["Streams"])
app.include_router(anpr.router,      prefix="/api/v1/anpr",      tags=["ANPR"])
app.include_router(biometric.router, prefix="/api/v1/biometric", tags=["Biometric"])
app.include_router(safety.router,    prefix="/api/v1/safety",    tags=["Safety"])
app.include_router(material.router,  prefix="/api/v1/material",  tags=["Material"])
app.include_router(asset.router,     prefix="/api/v1/asset",     tags=["Asset"])
app.include_router(vehicles.router,  prefix="/api/v1/vehicles",  tags=["Vehicles"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
