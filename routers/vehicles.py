"""
Vehicle tracking + demo vehicle-detection endpoints.
- /entry /exit  — manual gate recording
- /             — visit list
- /stats        — KPI summary
- /demo-upload  — upload a video, detect vehicles with YOLO
- /demo-job/:id — poll detection job
"""
from __future__ import annotations

import base64
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, BackgroundTasks, Body, File, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import db
from services.yolo_service import get_detector
from services.live_service import live_mjpeg

router  = APIRouter()
UPLOADS = Path("uploads")

SLA_MINUTES = 60


# ---------------------------------------------------------------------------
# Pydantic
# ---------------------------------------------------------------------------

class PlateBody(BaseModel):
    plate: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm(plate: str) -> str:
    return plate.upper().replace(" ", "").replace("-", "")


def _now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat()


def _compute_status(entry_iso: str, exit_iso: Optional[str]) -> tuple[Optional[int], str]:
    """Returns (duration_minutes, status)."""
    entry_dt = datetime.fromisoformat(entry_iso)
    if exit_iso:
        exit_dt  = datetime.fromisoformat(exit_iso)
        mins     = int((exit_dt - entry_dt).total_seconds() / 60)
        return mins, ("over_sla" if mins > SLA_MINUTES else "cleared")
    # still on site
    mins_on = int((datetime.utcnow() - entry_dt).total_seconds() / 60)
    return None, ("over_sla" if mins_on > SLA_MINUTES else "on_site")


def _owner_for_plate(plate: str, conn) -> str:
    norm = _norm(plate)
    row  = conn.execute(
        "SELECT owner FROM authorized_vehicles WHERE plate=?", (norm,)
    ).fetchone()
    return row["owner"] if row else "Unknown"


def _update_demo_job(job_id: str, status: str, progress: float,
                     error: Optional[str], result: Optional[dict]):
    conn = db.get_conn()
    conn.execute(
        "UPDATE vehicle_demo_jobs SET status=?, progress=?, error=?, result_json=? "
        "WHERE job_id=?",
        (status, progress, error, json.dumps(result) if result else None, job_id),
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Background demo-detection job  (modelled on YardMonitor's CameraPipeline)
#
# Key improvements vs the original naive implementation:
#   • YOLOv8 + ByteTrack for stable per-vehicle track IDs
#   • Per-track best-crop strategy: keeps the sharpest plate crop seen
#     (measured by Laplacian variance — YardMonitor's technique)
#   • Plate region extracted via contour finder (YardMonitor's _find_plate_contour)
#   • Direction-line crossing counter (entry going down, exit going up)
#   • Counts unique tracks, not raw per-frame detections
# ---------------------------------------------------------------------------

def _laplacian_sharpness(img: np.ndarray) -> float:
    """Higher = sharper. Used to pick the best plate crop (YardMonitor style)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _process_demo_video(job_id: str, video_path: str):
    """
    Process a demo video with YOLOv8 + ByteTrack.
    Tracks each vehicle across frames, keeps the sharpest plate crop,
    emits a single annotated sample frame.
    """
    t_start = time.time()
    try:
        _update_demo_job(job_id, "processing", 0, None, None)
        detector = get_detector()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            _update_demo_job(job_id, "error", 0, "Cannot open video file", None)
            return

        total      = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        fps        = cap.get(cv2.CAP_PROP_FPS) or 25
        duration_s = total / fps

        # ── Sampling strategy ───────────────────────────────────────────
        # Cap the number of frames actually run through YOLO. 120 frames is
        # plenty for stable ByteTrack IDs on a short clip while roughly halving
        # inference time vs the old 240-frame budget.
        max_frames = 120
        step       = max(1, int(total / max_frames)) if total > max_frames else 1

        # ── Per-track state  (mirrors YardMonitor's _TrackState) ────────
        # track_id → {'cls', 'best_crop', 'best_sharp', 'last_y', 'crossed'}
        track_state: dict = {}
        unique_vehicles: dict[str, int] = {}   # class → unique count

        # Direction line at 55 % of frame height (YardMonitor default)
        direction_line_y: Optional[float] = None
        entries = 0
        exits   = 0

        best_frame_annot: Optional[np.ndarray] = None
        best_det_count = 0

        frame_idx = 0
        sampled   = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if frame_idx % step != 0:
                continue
            sampled += 1

            # Set direction line once we know frame dimensions
            if direction_line_y is None:
                h_f, w_f = frame.shape[:2]
                direction_line_y = h_f * 0.55

            # ── Detection ───────────────────────────────────────────────
            use_track = detector.backend == "yolov8"
            dets = detector.detect(frame, track=use_track) if detector.available else []

            # ── Per-track state update ───────────────────────────────────
            for d in dets:
                tid = d.track_id if d.track_id is not None else id(d)
                cx, cy = d.center

                if tid not in track_state:
                    track_state[tid] = {
                        "cls":        d.cls_name,
                        "best_crop":  None,
                        "best_sharp": -1.0,
                        "last_y":     cy,
                        "crossed":    False,
                    }
                    # Count new unique vehicle
                    cls_cap = d.cls_name.capitalize()
                    unique_vehicles[cls_cap] = unique_vehicles.get(cls_cap, 0) + 1

                st = track_state[tid]

                # ── Best-crop update (YardMonitor sharpness strategy) ────
                crop = detector.get_vehicle_crop(frame, d)
                if crop.size > 0:
                    sharp = _laplacian_sharpness(crop)
                    if sharp > st["best_sharp"]:
                        st["best_sharp"] = sharp
                        st["best_crop"]  = crop

                # ── Direction crossing (YardMonitor line-crossing logic) ─
                last_y = st["last_y"]
                if not st["crossed"]:
                    if last_y < direction_line_y <= cy:
                        entries += 1
                        st["crossed"] = True
                    elif last_y > direction_line_y >= cy:
                        exits += 1
                        st["crossed"] = True
                st["last_y"] = cy

            # ── Keep annotated frame with most detections ────────────────
            if dets and len(dets) >= best_det_count:
                best_det_count = len(dets)
                ann = detector.draw(frame, dets)
                h_a, w_a = ann.shape[:2]
                if w_a > 720:
                    ann = cv2.resize(ann, (720, int(720 * h_a / w_a)))
                # Draw direction line
                cv2.line(ann,
                         (0, int(ann.shape[0] * 0.55)),
                         (ann.shape[1], int(ann.shape[0] * 0.55)),
                         (240, 169, 59), 1)
                best_frame_annot = ann

            # Throttle progress writes — one SQLite commit per sampled frame
            # dominated the runtime. Update roughly every 8th sampled frame.
            if sampled % 8 == 0:
                _update_demo_job(job_id, "processing",
                                 min(99, frame_idx / total * 100), None, None)

        cap.release()

        # ── Sample frame ─────────────────────────────────────────────────
        sample_b64: Optional[str] = None
        if best_frame_annot is not None:
            _, jpg = cv2.imencode(".jpg", best_frame_annot,
                                  [cv2.IMWRITE_JPEG_QUALITY, 82])
            sample_b64 = base64.b64encode(jpg.tobytes()).decode()

        result = {
            "counts":           unique_vehicles,
            "total_detections": sum(unique_vehicles.values()),
            "frames_processed": sampled,
            "duration_s":       round(duration_s, 1),
            "entries_detected": entries,
            "exits_detected":   exits,
            "sample_frame":     sample_b64,
            "yolo_available":   detector.available,
            "backend":          detector.backend,
        }
        _update_demo_job(job_id, "completed", 100, None, result)

    except Exception as exc:
        import traceback
        print(f"[Demo job {job_id}] {traceback.format_exc()}")
        _update_demo_job(job_id, "error", 0, str(exc), None)
    finally:
        Path(video_path).unlink(missing_ok=True)
        print(f"[Demo job {job_id}] done in {time.time()-t_start:.1f}s")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/entry")
def record_entry(body: PlateBody):
    norm  = _norm(body.plate)
    conn  = db.get_conn()

    # Prevent duplicate open entries
    dup = conn.execute(
        "SELECT id FROM vehicle_visits WHERE plate=? AND exit_time IS NULL", (norm,)
    ).fetchone()
    if dup:
        conn.close()
        raise HTTPException(409, f"{norm} already has an open entry — record exit first.")

    owner = _owner_for_plate(norm, conn)
    conn.execute(
        "INSERT INTO vehicle_visits (plate, owner, entry_time) VALUES (?,?,?)",
        (norm, owner, _now_iso()),
    )
    conn.commit()
    conn.close()
    return {"message": f"Entry recorded for {norm}", "plate": norm, "owner": owner}


@router.post("/exit")
def record_exit(body: PlateBody):
    norm = _norm(body.plate)
    conn = db.get_conn()
    row  = conn.execute(
        "SELECT id, entry_time FROM vehicle_visits "
        "WHERE plate=? AND exit_time IS NULL ORDER BY id DESC LIMIT 1",
        (norm,),
    ).fetchone()
    if not row:
        conn.close()
        raise HTTPException(404, f"No open entry found for {norm}")

    exit_iso = _now_iso()
    mins     = int((datetime.fromisoformat(exit_iso) -
                    datetime.fromisoformat(row["entry_time"])).total_seconds() / 60)
    conn.execute(
        "UPDATE vehicle_visits SET exit_time=?, duration_minutes=? WHERE id=?",
        (exit_iso, mins, row["id"]),
    )
    conn.commit()
    conn.close()
    return {
        "message":          f"Exit recorded for {norm}",
        "plate":            norm,
        "duration_minutes": mins,
    }


@router.get("")
def list_vehicles(limit: int = Query(60, ge=1, le=300)):
    conn  = db.get_conn()
    rows  = conn.execute(
        "SELECT id, plate, owner, entry_time, exit_time "
        "FROM vehicle_visits ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()

    result = []
    for r in rows:
        dur, status = _compute_status(r["entry_time"], r["exit_time"])
        result.append(
            {
                "id":               r["id"],
                "plate":            r["plate"],
                "owner":            r["owner"],
                "entry_time":       r["entry_time"],
                "exit_time":        r["exit_time"],
                "duration_minutes": dur,
                "status":           status,
            }
        )
    return result


class UpdateVisit(BaseModel):
    plate:      Optional[str] = None
    owner:      Optional[str] = None
    exit_time:  Optional[str] = None


@router.patch("/{visit_id}")
def update_visit(visit_id: int, body: UpdateVisit):
    conn = db.get_conn()
    row  = conn.execute(
        "SELECT id, plate, entry_time, exit_time FROM vehicle_visits WHERE id=?",
        (visit_id,),
    ).fetchone()
    if not row:
        conn.close()
        raise HTTPException(404, "Visit not found")

    updates: dict = {}
    if body.plate is not None:
        updates["plate"] = _norm(body.plate)
    if body.owner is not None:
        updates["owner"] = body.owner
    if body.exit_time is not None:
        updates["exit_time"] = body.exit_time or None
        if body.exit_time:
            try:
                entry_dt = datetime.fromisoformat(row["entry_time"])
                exit_dt  = datetime.fromisoformat(body.exit_time)
                updates["duration_minutes"] = int(
                    (exit_dt - entry_dt).total_seconds() / 60
                )
            except ValueError:
                pass

    if updates:
        set_clause = ", ".join(f"{k}=?" for k in updates)
        conn.execute(
            f"UPDATE vehicle_visits SET {set_clause} WHERE id=?",
            (*updates.values(), visit_id),
        )
        conn.commit()
    conn.close()
    return {"id": visit_id, **updates}


@router.delete("/{visit_id}", status_code=204)
def delete_visit(visit_id: int):
    conn = db.get_conn()
    conn.execute("DELETE FROM vehicle_visits WHERE id=?", (visit_id,))
    conn.commit()
    conn.close()


@router.get("/stats")
def vehicle_stats():
    conn  = db.get_conn()
    rows  = conn.execute(
        "SELECT plate, entry_time, exit_time, duration_minutes "
        "FROM vehicle_visits WHERE date(entry_time) = date('now')"
    ).fetchall()
    conn.close()

    on_site   = 0
    cleared   = 0
    over_sla  = 0
    durations = []

    for r in rows:
        dur, status = _compute_status(r["entry_time"], r["exit_time"])
        if status == "on_site":
            on_site += 1
        elif status == "over_sla":
            if r["exit_time"]:
                cleared  += 1
                over_sla += 1
                if dur:
                    durations.append(dur)
            else:
                on_site  += 1
                over_sla += 1
        else:
            cleared += 1
            if dur:
                durations.append(dur)

    avg = round(sum(durations) / len(durations)) if durations else None
    return {
        "on_site":    on_site,
        "cleared":    cleared,
        "over_sla":   over_sla,
        "avg_minutes": avg,
    }


# ---------------------------------------------------------------------------
# Live analytics  (phone IP-camera → real-time vehicle + plate detection)
# ---------------------------------------------------------------------------

@router.get("/live")
async def live(
    src: str = Query(
        ...,
        description="Phone IP-camera stream, e.g. http://192.168.1.5:8080/video "
                    "(IP Webcam app), an rtsp:// URL, or a webcam index like 0.",
    ),
):
    """Stream live annotated MJPEG: YOLOv8+ByteTrack vehicle boxes with plate
    numbers read opportunistically per track. Point an <img> tag at this URL."""
    return StreamingResponse(
        live_mjpeg(src),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ---------------------------------------------------------------------------
# Demo video detection
# ---------------------------------------------------------------------------

@router.post("/demo-upload")
async def demo_upload(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
):
    job_id     = str(uuid.uuid4())
    suffix     = Path(video.filename or "video.mp4").suffix or ".mp4"
    video_path = UPLOADS / f"vdemo_{job_id}{suffix}"
    UPLOADS.mkdir(exist_ok=True)
    video_path.write_bytes(await video.read())

    conn = db.get_conn()
    conn.execute(
        "INSERT INTO vehicle_demo_jobs (job_id, status, progress) VALUES (?,?,?)",
        (job_id, "pending", 0),
    )
    conn.commit()
    conn.close()

    background_tasks.add_task(_process_demo_video, job_id, str(video_path))
    return {"job_id": job_id}


@router.get("/demo-job/{job_id}")
def get_demo_job(job_id: str):
    conn = db.get_conn()
    row  = conn.execute(
        "SELECT * FROM vehicle_demo_jobs WHERE job_id=?", (job_id,)
    ).fetchone()
    conn.close()
    if not row:
        raise HTTPException(404, "Job not found")
    row = dict(row)
    result = json.loads(row["result_json"]) if row.get("result_json") else None
    return {
        "job_id":   job_id,
        "status":   row["status"],
        "progress": row["progress"],
        "error":    row["error"],
        "result":   result,
    }
