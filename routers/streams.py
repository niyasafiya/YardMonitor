"""
MJPEG camera-stream endpoints.

Each camera can stream from a REAL CCTV via RTSP, or fall back to a synthetic
placeholder frame when no RTSP source is configured.

To connect a real camera, set an environment variable named <CAMID>_RTSP before
starting the backend, e.g. (PowerShell):

    $env:CAM01_RTSP = "rtsp://admin:PASSWORD@192.168.1.64:554/Streaming/Channels/102"
    python main.py

or add the same line to a `.env` file (see cameras.env.example). If the variable
is not set, that camera keeps showing the "NO CAMERA FEED" placeholder.
"""
from __future__ import annotations

import asyncio
import os
import time
import cv2
import numpy as np
from fastapi import APIRouter, Response
from fastapi.responses import StreamingResponse

router = APIRouter()

_CAM_CFG = {
    "cam01": {"name": "Main Gate — In",     "id": "CAM-01", "analytics": "LPR · Face"},
    "cam02": {"name": "Facial Recog - Gate", "id": "CAM-02", "analytics": "Face ID"},
    "cam12": {"name": "Loading Bay 2",      "id": "CAM-12", "analytics": "DN-OCR · Count"},
    "cam08": {"name": "Warehouse Aisle 4",  "id": "CAM-08", "analytics": "PPE · Zone"},
    "cam05": {"name": "Turnstile B",        "id": "CAM-05", "analytics": "Tailgate"},
}


def _rtsp_url(cam_id: str) -> str | None:
    """Return the configured RTSP URL for a camera, or None for synthetic mode.

    Looks up an env var named e.g. CAM01_RTSP (upper-cased cam id + _RTSP).
    """
    return os.environ.get(f"{cam_id.upper()}_RTSP") or None


def _make_frame(cam_id: str) -> bytes:
    cfg = _CAM_CFG.get(cam_id, {"name": "Unknown", "id": cam_id, "analytics": ""})
    h, w = 360, 640
    frame = np.full((h, w, 3), [10, 16, 22], dtype=np.uint8)

    # Dim grid
    for x in range(0, w, 64):
        cv2.line(frame, (x, 0), (x, h), (16, 24, 32), 1)
    for y in range(0, h, 64):
        cv2.line(frame, (0, y), (w, y), (16, 24, 32), 1)

    # Animated scan-line
    t   = time.time()
    sy  = int((t % 3.5) / 3.5 * h)
    cv2.line(frame, (0, sy), (w, sy), (0, 180, 140), 1)

    # Top-left: green dot + camera name + ID
    cv2.circle(frame, (14, 14), 5, (0, 200, 155), -1)
    cv2.putText(frame, cfg["name"], (26, 19),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (195, 215, 235), 1)
    cv2.putText(frame, cfg["id"],   (26, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (80, 130, 170), 1)

    # Top-right: timestamp + LIVE
    ts = time.strftime("%H:%M:%S")
    cv2.putText(frame, ts,      (w - 72, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (170, 200, 220), 1)
    cv2.putText(frame, "LIVE",  (w - 40, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 200, 155), 1)

    # Centre message
    for txt, yp, sc, col in [
        ("NO CAMERA FEED",                      h // 2 - 22, 0.62, (140, 170, 195)),
        ("Connect RTSP / upload demo video",    h // 2 +  4, 0.38, ( 70, 115, 148)),
        (f"Analytics: {cfg['analytics']}",      h // 2 + 20, 0.36, ( 55, 100, 135)),
    ]:
        tw = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, sc, 1)[0][0]
        cv2.putText(frame, txt, ((w - tw) // 2, yp),
                    cv2.FONT_HERSHEY_SIMPLEX, sc, col, 1)

    # Bottom bar
    cv2.rectangle(frame, (0, h - 28), (w, h), (8, 13, 20), -1)
    cv2.putText(frame,
                f"Sentinel AI · Warehouse North · 1080p · {time.strftime('%d %b %Y')}",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (55, 95, 130), 1)

    _, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return jpg.tobytes()


def _status_frame(cam_id: str, line1: str, line2: str = "") -> bytes:
    """A dark placeholder frame carrying a status message (e.g. connecting/offline)."""
    cfg = _CAM_CFG.get(cam_id, {"name": "Unknown", "id": cam_id, "analytics": ""})
    h, w = 360, 640
    frame = np.full((h, w, 3), [10, 16, 22], dtype=np.uint8)
    cv2.circle(frame, (14, 14), 5, (0, 140, 200), -1)
    cv2.putText(frame, cfg["name"], (26, 19),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (195, 215, 235), 1)
    cv2.putText(frame, cfg["id"], (26, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (80, 130, 170), 1)
    for txt, yp, sc, col in [
        (line1, h // 2 - 6, 0.60, (150, 180, 205)),
        (line2, h // 2 + 20, 0.38, (70, 115, 148)),
    ]:
        if not txt:
            continue
        tw = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, sc, 1)[0][0]
        cv2.putText(frame, txt, ((w - tw) // 2, yp),
                    cv2.FONT_HERSHEY_SIMPLEX, sc, col, 1)
    _, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return jpg.tobytes()


def _open_capture(url: str) -> cv2.VideoCapture:
    """Open an RTSP stream, preferring TCP transport (more reliable than UDP)."""
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # keep latency low
    except Exception:
        pass
    return cap


async def _rtsp_generator(cam_id: str, url: str):
    """Stream live frames from a real CCTV. Auto-reconnects if the link drops."""
    boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
    cap: cv2.VideoCapture | None = None
    fail = 0
    while True:
        if cap is None or not cap.isOpened():
            if cap is not None:
                cap.release()
            yield boundary + _status_frame(cam_id, "Connecting to camera…", cam_id.upper()) + b"\r\n"
            cap = await asyncio.to_thread(_open_capture, url)
            if not cap.isOpened():
                fail += 1
                yield boundary + _status_frame(
                    cam_id, "CAMERA OFFLINE", "Check RTSP URL / network") + b"\r\n"
                await asyncio.sleep(min(1.0 * fail, 5.0))   # backoff, cap at 5s
                continue
            fail = 0

        ok, frame = await asyncio.to_thread(cap.read)
        if not ok or frame is None:
            cap.release()
            cap = None
            await asyncio.sleep(0.3)
            continue

        # Normalise size so the console tiles stay consistent
        if frame.shape[1] != 640:
            h = int(frame.shape[0] * 640 / frame.shape[1])
            frame = cv2.resize(frame, (640, h))
        ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            yield boundary + jpg.tobytes() + b"\r\n"
        await asyncio.sleep(0.04)   # ~25 fps ceiling; real rate follows the camera


async def _mjpeg_generator(cam_id: str):
    """Synthetic placeholder stream (used when no RTSP source is configured)."""
    boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
    while True:
        jpg = _make_frame(cam_id)
        yield boundary + jpg + b"\r\n"
        await asyncio.sleep(0.10)   # ~10 fps


@router.get("/stream/{cam_id}")
async def stream(cam_id: str):
    if cam_id not in _CAM_CFG:
        return Response(status_code=404, content="Unknown camera")
    url = _rtsp_url(cam_id)
    gen = _rtsp_generator(cam_id, url) if url else _mjpeg_generator(cam_id)
    return StreamingResponse(
        gen,
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
