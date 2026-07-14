"""
Asset tracking endpoints — recognise the assets on a truck and match them to a
register of known company assets.

Upload a photo / clip of the truck bed (or stream single frames from a live
webcam) — the service detects each asset, names its category, and this router
matches those categories against the editable `assets` register to surface the
real asset tag (e.g. GEN-042 · Diesel Generator). Every detection is written to
asset_log for the tracking audit trail.

Mounted at /api/v1/asset.
"""
from __future__ import annotations

import asyncio
import base64
import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel

import db
from services import asset_service as as_
from services import material_service as ms

log = logging.getLogger(__name__)
router = APIRouter()

_IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
_VID_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


# ---------------------------------------------------------------------------
# Register helpers
# ---------------------------------------------------------------------------

def _registered() -> List[dict]:
    conn = db.get_conn()
    rows = conn.execute(
        "SELECT asset_id, name, category, note FROM assets ORDER BY asset_id"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def _match(categories: dict) -> dict:
    """
    Map detected categories -> registered assets sharing that category.

    Returns:
      items          – per detected category: count + matched register entries
      matched_assets – flat list of registered assets seen in this frame
      label_by_cat   – category -> annotation label for the richest match
    """
    reg = _registered()
    by_cat: Dict[str, List[dict]] = {}
    for a in reg:
        by_cat.setdefault(as_.normalise(a["category"]), []).append(a)

    items: List[dict] = []
    matched_assets: List[dict] = []
    label_by_cat: Dict[str, str] = {}
    for cat, count in sorted(categories.items()):
        matches = by_cat.get(cat, [])
        items.append({
            "category": cat,
            "count": count,
            "registered": bool(matches),
            "matches": [{"asset_id": m["asset_id"], "name": m["name"]} for m in matches],
        })
        if matches:
            matched_assets.extend(
                {"asset_id": m["asset_id"], "name": m["name"], "category": cat}
                for m in matches
            )
            first = matches[0]
            extra = f" +{len(matches) - 1}" if len(matches) > 1 else ""
            label_by_cat[cat] = f"{first['asset_id']} - {first['name']}{extra}"
    return {"items": items, "matched_assets": matched_assets, "label_by_cat": label_by_cat}


def _log_detection(plate: str, source: str, detected: List[str],
                   matched: List[dict], item_count: int, location: str = ""):
    try:
        conn = db.get_conn()
        conn.execute(
            "INSERT INTO asset_log (plate, source, detected, matched, item_count, location) "
            "VALUES (?,?,?,?,?,?)",
            (plate or "", source or "upload", ", ".join(detected),
             ", ".join(m["asset_id"] for m in matched), int(item_count), location or ""),
        )
        conn.commit()
        conn.close()
    except Exception:
        log.exception("asset_log write failed")


# ---------------------------------------------------------------------------
# Core analysis (worker thread)
# ---------------------------------------------------------------------------

def _analyze_frame(frame) -> dict:
    if not as_.available():
        raise RuntimeError("Asset detection model unavailable (ultralytics/model not loaded)")

    analysis = as_.analyze_frame(frame)
    categories = dict(analysis["categories"])

    # Custom load classifier (if trained) recognises bulk/wrapped cargo the
    # generic detector misses — e.g. a truck of mattresses. A confident result
    # is folded in as a whole-load "asset" so it matches the register & logs.
    load = as_.classify_load(frame)
    load_type = None
    if load and load["confident"]:
        load_type = load
        categories.setdefault(load["label"], 0)
        if categories[load["label"]] == 0:
            categories[load["label"]] = 1          # count the load as one asset

    m = _match(categories)

    annotated = as_.draw_boxes(frame, analysis["boxes"], m["label_by_cat"])
    if load_type:
        reg = m["label_by_cat"].get(load["label"])
        banner = (f"LOAD: {reg}" if reg
                  else f"LOAD: {load['label']}") + f"  {load['conf']*100:.0f}%"
        annotated = as_.draw_load_banner(annotated, banner, bool(reg))
    ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 88])
    img_b64 = ("data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()
               if ok else None)

    return {
        "status": "complete",
        "detected": sorted(categories.keys()),
        "items": m["items"],
        "matched_assets": m["matched_assets"],
        "item_count": sum(categories.values()),
        "load_type": load_type,          # {label, conf, confident} or None
        "annotated_image": img_b64,
    }


# Debug capture: set ASSET_DEBUG=1 to save the last uploaded frame + a dump of
# EVERY raw detection at a very low threshold to data/asset_debug/, so a
# difficult image (e.g. mattresses on a truck) can be diagnosed. Off by default
# — it writes files and runs an extra low-confidence inference per upload.
_DEBUG = os.environ.get("ASSET_DEBUG", "0") == "1"


def _debug_capture(frame, tag: str = "upload"):
    if not _DEBUG:
        return
    try:
        out = Path("data/asset_debug")
        out.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out / "last_upload.jpg"), frame)
        det = ms_raw()
        lines = [f"tag={tag}  frame={frame.shape[1]}x{frame.shape[0]}"]
        if det is not None:
            res = det(frame, conf=0.05, verbose=False, imgsz=1280)[0]
            names = res.names
            rows = sorted(
                ((names[int(c)], float(cf)) for c, cf in zip(res.boxes.cls, res.boxes.conf)),
                key=lambda x: -x[1],
            )
            lines.append(f"raw detections (conf>=0.05): {len(rows)}")
            lines += [f"  {n:16s} {cf:.3f}" for n, cf in rows]
        (out / "last_upload.txt").write_text("\n".join(lines), encoding="utf-8")
        log.info("[Asset][debug] captured %s — %d raw dets", tag, len(rows) if det else 0)
    except Exception:
        log.exception("asset debug capture failed")


def ms_raw():
    """Underlying YOLO model (for raw low-conf debug), or None."""
    det = ms.get_material_detector()
    return det._model if det.available else None


def _run_image(content: bytes) -> dict:
    arr = np.frombuffer(content, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode image — unsupported format?")
    _debug_capture(frame, "image")
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

        if not as_.available():
            raise RuntimeError("Asset detection model unavailable")

        # Aggregate assets across the clip; keep the richest frame to annotate.
        agg: Dict[str, int] = {}
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
                if fw > 1280:                 # keep detail for hard items (bed on truck)
                    frame = cv2.resize(frame, (1280, int(fh * 1280 / fw)))
                a = as_.analyze_frame(frame)
                sampled += 1
                for c, n in a["categories"].items():
                    agg[c] = max(agg.get(c, 0), n)
                if len(a["boxes"]) > best_count:
                    best_count = len(a["boxes"])
                    best_frame = frame.copy()
            idx += 1
        cap.release()

        if best_frame is None:
            return {"status": "complete", "detected": [], "items": [],
                    "matched_assets": [], "item_count": 0, "annotated_image": None}

        _debug_capture(best_frame, "video")

        # Annotate the best frame but report the aggregate over the whole clip.
        result = _analyze_frame(best_frame)
        # Fold the whole-load classification (from the best frame) into the
        # clip aggregate so the reported items/decision include it too.
        if result.get("load_type") and result["load_type"]["confident"]:
            agg.setdefault(result["load_type"]["label"], 0)
            if agg[result["load_type"]["label"]] == 0:
                agg[result["load_type"]["label"]] = 1
        m = _match(agg)
        result["detected"] = sorted(agg.keys())
        result["items"] = m["items"]
        result["matched_assets"] = m["matched_assets"]
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
    clf = as_.get_load_classifier()
    return {
        "available": as_.available(),
        "registered_count": len(_registered()),
        "custom_model": clf is not None,
        "custom_classes": sorted(as_.normalise(n) for n in clf.names.values()) if clf else [],
    }


@router.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...), plate: str = Form(""),
                        source: str = Form("upload")):
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
        log.exception("Asset image analysis failed")
        raise HTTPException(500, f"Analysis error: {e}")

    if result["item_count"]:
        _log_detection(plate, source, result["detected"],
                       result["matched_assets"], result["item_count"])
    return result


@router.post("/analyze-video")
async def analyze_video(file: UploadFile = File(...), plate: str = Form(""),
                        source: str = Form("upload")):
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
        log.exception("Asset video analysis failed")
        raise HTTPException(500, f"Analysis error: {e}")

    if result["item_count"]:
        _log_detection(plate, source, result["detected"],
                       result["matched_assets"], result["item_count"])
    return result


@router.post("/analyze-frame")
async def analyze_frame(photo: UploadFile = File(...), log_it: bool = Query(False, alias="log"),
                        plate: str = Form(""), source: str = Form("live")):
    """Single live frame — fast detection for the live feed. Logs only when log=true."""
    content = await photo.read()
    try:
        result = await asyncio.to_thread(_run_image, content)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    except Exception as e:
        log.exception("Asset frame analysis failed")
        raise HTTPException(500, f"Analysis error: {e}")

    if log_it and result["item_count"]:
        _log_detection(plate, source, result["detected"],
                       result["matched_assets"], result["item_count"])
    return result


# ---------------------------------------------------------------------------
# Register CRUD
# ---------------------------------------------------------------------------

class AssetIn(BaseModel):
    asset_id: str
    name: str
    category: str = ""
    note: str = ""


@router.get("/assets")
def list_assets():
    conn = db.get_conn()
    rows = conn.execute(
        "SELECT asset_id, name, category, note, created_at FROM assets ORDER BY asset_id"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@router.post("/assets", status_code=201)
def add_asset(body: AssetIn):
    asset_id = (body.asset_id or "").strip().upper()
    name = (body.name or "").strip()
    if not asset_id or not name:
        raise HTTPException(400, "Asset tag and name are required")
    category = as_.normalise(body.category)
    conn = db.get_conn()
    conn.execute(
        "INSERT OR REPLACE INTO assets (asset_id, name, category, note) VALUES (?,?,?,?)",
        (asset_id, name, category, body.note or ""),
    )
    conn.commit()
    conn.close()
    return {"asset_id": asset_id, "name": name, "category": category, "note": body.note or ""}


@router.delete("/assets/{asset_id}", status_code=204)
def delete_asset(asset_id: str):
    conn = db.get_conn()
    conn.execute("DELETE FROM assets WHERE asset_id=?", ((asset_id or "").strip().upper(),))
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Tracking log
# ---------------------------------------------------------------------------

@router.post("/log", status_code=201)
def add_log(plate: str = Form(""), source: str = Form("live"),
            detected: str = Form(""), matched: str = Form(""),
            item_count: int = Form(0), location: str = Form("")):
    """Record a tracking event (used by the live feed on a change of detection)."""
    conn = db.get_conn()
    conn.execute(
        "INSERT INTO asset_log (plate, source, detected, matched, item_count, location) "
        "VALUES (?,?,?,?,?,?)",
        (plate, source, detected, matched, int(item_count), location),
    )
    conn.commit()
    conn.close()
    return {"status": "logged"}


@router.get("/log")
def asset_log(limit: int = Query(30, ge=1, le=200)):
    conn = db.get_conn()
    rows = conn.execute(
        "SELECT id, timestamp, plate, source, detected, matched, item_count, location "
        "FROM asset_log ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@router.delete("/log/{log_id}", status_code=204)
def delete_asset_log(log_id: int):
    conn = db.get_conn()
    conn.execute("DELETE FROM asset_log WHERE id=?", (log_id,))
    conn.commit()
    conn.close()


@router.delete("/log", status_code=204)
def clear_asset_log():
    conn = db.get_conn()
    conn.execute("DELETE FROM asset_log")
    conn.commit()
    conn.close()
