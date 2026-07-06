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
from routers import anpr, biometric, safety, vehicles, streams

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
    from services.anpr_service import _get_paddle, _get_easyocr
    print("[Startup] Loading OCR models in background…")
    reader = _get_paddle()
    if reader is None or reader == "FAILED":
        _get_easyocr()
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
app.include_router(vehicles.router,  prefix="/api/v1/vehicles",  tags=["Vehicles"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
