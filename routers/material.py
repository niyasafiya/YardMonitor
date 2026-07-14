"""
Material & DN endpoints — cargo inspection + allow-list access control.

Upload an image or short clip of what a vehicle is carrying. The service
detects the items inside, maps them to materials, and this router compares them
against the editable allowed_materials list to GRANT or DENY yard entry, logging
each decision to material_log for the audit trail.

Mounted at /api/v1/material.
"""
from __future__ import annotations

import asyncio
import base64
import logging
import os
import tempfile
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel

import db
from services import material_service as ms

log = logging.getLogger(__name__)
router = APIRouter()

_IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
_VID_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


# ---------------------------------------------------------------------------
# Allow-list helpers
# ---------------------------------------------------------------------------

def _allowed_set() -> set:
    conn = db.get_conn()
    rows = conn.execute("SELECT material FROM allowed_materials").fetchall()
    conn.close()
    return {r["material"] for r in rows}


def _decide(materials: dict) -> dict:
    """Compare detected materials against the allow-list → GRANT / DENY."""
    allowed = _allowed_set()
    detected = sorted(materials.keys())
    disallowed = [m for m in detected if m not in allowed]
    granted = len(detected) > 0 and not disallowed
    # Empty load: nothing detected — treat as manual review, not an auto-grant.
    if not detected:
        decision = "REVIEW"
    else:
        decision = "GRANTED" if granted else "DENIED"
    return {
        "decision": decision,
        "detected": detected,
        "disallowed": disallowed,
        "allowed_list": sorted(allowed),
    }


def _log_decision(plate: str, verdict: dict, location: str = ""):
    try:
        conn = db.get_conn()
        conn.execute(
            "INSERT INTO material_log (plate, detected, disallowed, decision, location) "
            "VALUES (?,?,?,?,?)",
            (plate or "", ", ".join(verdict["detected"]),
             ", ".join(verdict["disallowed"]), verdict["decision"], location or ""),
        )
        conn.commit()
        conn.close()
    except Exception:
        log.exception("material_log write failed")


# ---------------------------------------------------------------------------
# Core analysis (worker thread)
# ---------------------------------------------------------------------------

def _analyze_frame(frame) -> dict:
    det = ms.get_material_detector()
    if not det.available:
        raise RuntimeError("Material detection model unavailable (ultralytics/model not loaded)")

    analysis = det.analyze_frame(frame)
    verdict = _decide(analysis["materials"])

    allowed = _allowed_set()
    annotated = ms.draw_boxes(frame, analysis["boxes"], allowed)
    ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 88])
    img_b64 = ("data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()
               if ok else None)

    items = [
        {"material": m, "count": c, "allowed": m in allowed}
        for m, c in sorted(analysis["materials"].items())
    ]
    return {
        "status": "complete",
        "decision": verdict["decision"],
        "detected": verdict["detected"],
        "disallowed": verdict["disallowed"],
        "items": items,
        "item_count": sum(analysis["materials"].values()),
        "annotated_image": img_b64,
    }


def _run_image(content: bytes) -> dict:
    arr = np.frombuffer(content, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode image — unsupported format?")
    return _analyze_frame(frame)


def _run_video(content: bytes, suffix: str) -> dict:
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise ValueError("Could not open video file — unsupported format?")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        max_frames = min(total or int(fps * 30), int(fps * 30))   # ≤ 30 s
        step = max(1, int(fps))                                   # ~1 fps
        MAX_SAMPLES = 15

        det = ms.get_material_detector()
        if not det.available:
            raise RuntimeError("Material detection model unavailable")

        # Aggregate materials across the clip; keep the richest frame to annotate.
        agg: dict = {}
        best_frame = None
        best_count = -1
        idx = 0
        sampled = 0
        while idx < max_frames and sampled < MAX_SAMPLES:
            if not cap.grab():
                break
            if idx % step == 0:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                fh, fw = frame.shape[:2]
                if fw > 960:
                    frame = cv2.resize(frame, (960, int(fh * 960 / fw)))
                a = det.analyze_frame(frame)
                sampled += 1
                for m, c in a["materials"].items():
                    agg[m] = max(agg.get(m, 0), c)
                if len(a["boxes"]) > best_count:
                    best_count = len(a["boxes"])
                    best_frame = frame.copy()
            idx += 1
        cap.release()

        if best_frame is None:
            return {"status": "complete", "decision": "REVIEW", "detected": [],
                    "disallowed": [], "items": [], "item_count": 0,
                    "annotated_image": None}

        # Annotate the best frame but decide on the aggregate of the whole clip.
        result = _analyze_frame(best_frame)
        verdict = _decide(agg)
        allowed = _allowed_set()
        result["decision"] = verdict["decision"]
        result["detected"] = verdict["detected"]
        result["disallowed"] = verdict["disallowed"]
        result["items"] = [
            {"material": m, "count": c, "allowed": m in allowed}
            for m, c in sorted(agg.items())
        ]
        result["item_count"] = sum(agg.values())
        return result
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Analyze routes
# ---------------------------------------------------------------------------

@router.get("/status")
def status():
    det = ms.get_material_detector()
    return {"available": det.available, "allowed_count": len(_allowed_set())}


@router.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...), plate: str = Form("")):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in _IMG_EXT:
        raise HTTPException(400, f"Unsupported image type '{ext}'. Allowed: {', '.join(sorted(_IMG_EXT))}")
    content = await file.read()
    if len(content) > 20 * 1024 * 1024:
        raise HTTPException(413, "Image too large (max 20 MB)")
    try:
        result = await asyncio.to_thread(_run_image, content)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    except Exception as e:
        log.exception("Material image analysis failed")
        raise HTTPException(500, f"Analysis error: {e}")

    _log_decision(plate, {"detected": result["detected"],
                          "disallowed": result["disallowed"],
                          "decision": result["decision"]})
    return result


@router.post("/analyze-video")
async def analyze_video(file: UploadFile = File(...), plate: str = Form("")):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in _VID_EXT:
        raise HTTPException(400, f"Unsupported video type '{ext}'. Allowed: {', '.join(sorted(_VID_EXT))}")
    content = await file.read()
    if len(content) > 500 * 1024 * 1024:
        raise HTTPException(413, "Video too large (max 500 MB)")
    try:
        result = await asyncio.to_thread(_run_video, content, ext)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    except Exception as e:
        log.exception("Material video analysis failed")
        raise HTTPException(500, f"Analysis error: {e}")

    _log_decision(plate, {"detected": result["detected"],
                          "disallowed": result["disallowed"],
                          "decision": result["decision"]})
    return result


# ---------------------------------------------------------------------------
# Allow-list CRUD
# ---------------------------------------------------------------------------

class MaterialIn(BaseModel):
    material: str
    label: str = ""
    note: str = ""


@router.get("/allowed")
def list_allowed():
    conn = db.get_conn()
    rows = conn.execute(
        "SELECT material, label, note, created_at FROM allowed_materials ORDER BY material"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@router.post("/allowed", status_code=201)
def add_allowed(body: MaterialIn):
    material = ms.normalise(body.material)
    if not material:
        raise HTTPException(400, "Material name required")
    conn = db.get_conn()
    conn.execute(
        "INSERT OR REPLACE INTO allowed_materials (material, label, note) VALUES (?,?,?)",
        (material, body.label or material.title(), body.note or ""),
    )
    conn.commit()
    conn.close()
    return {"material": material, "label": body.label or material.title(), "note": body.note or ""}


@router.delete("/allowed/{material}", status_code=204)
def delete_allowed(material: str):
    conn = db.get_conn()
    conn.execute("DELETE FROM allowed_materials WHERE material=?", (ms.normalise(material),))
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Decision log
# ---------------------------------------------------------------------------

@router.get("/log")
def material_log(limit: int = Query(30, ge=1, le=200)):
    conn = db.get_conn()
    rows = conn.execute(
        "SELECT id, timestamp, plate, detected, disallowed, decision, location "
        "FROM material_log ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@router.delete("/log/{log_id}", status_code=204)
def delete_material_log(log_id: int):
    conn = db.get_conn()
    conn.execute("DELETE FROM material_log WHERE id=?", (log_id,))
    conn.commit()
    conn.close()


@router.delete("/log", status_code=204)
def clear_material_log():
    conn = db.get_conn()
    conn.execute("DELETE FROM material_log")
    conn.commit()
    conn.close()
