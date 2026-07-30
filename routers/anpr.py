"""
ANPR (Automatic Number Plate Recognition) endpoints.
Video upload → background OCR job → annotated output video + access log.
"""
from __future__ import annotations

import asyncio
import base64
import difflib
import json
import tempfile
import threading
import uuid
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

import db
from services.anpr_service import (
    _looks_like_plate,
    _majority_vote,
    _postfix,
    extract_plates_from_frame,
)
from services.yolo_service import get_detector

router = APIRouter()

# Use system temp dir so files are NOT inside OneDrive (avoids cloud sync locking)
UPLOADS = Path(tempfile.gettempdir()) / "sentinel_ai"

# Maximum frames to annotate (~10 min @ 30 fps)
_ANNOTATE_MAX_FRAMES = 18_000

# Downscale frames wider than this before OCR/vehicle detection — big speed win on
# 1080p/4K clips, plates stay readable. Bboxes are scaled back to full resolution.
_MAX_SCAN_W = 1280


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class AuthVehicle(BaseModel):
    plate:        str
    owner:        str = "Unknown"
    vehicle_type: str = "Car"


class AuthVehicleUpdate(BaseModel):
    owner:        Optional[str] = None
    vehicle_type: Optional[str] = None
    new_plate:    Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm(plate: str) -> str:
    return plate.upper().replace(" ", "").replace("-", "")


def _cluster_and_vote(all_dets: List[dict]) -> dict:
    """Accuracy pass: group every plate read across all sampled frames by string
    similarity, then run a confidence-weighted character-level majority vote on
    each group. This corrects single-frame OCR mistakes (1/I, 0/O, S/5, B/8 …)
    by letting multiple frames agree on each character. No extra OCR work, so
    scan speed is unchanged.

    Returns {norm_plate: {"plate", "confidence", "bbox"}} and also rewrites each
    detection's "plate" in place so the annotated video matches the final read.
    """
    clusters: List[dict] = []
    for d in all_dets:
        p = _norm(d["plate"])
        if not p:
            continue
        placed = False
        for c in clusters:
            # Compare against the cluster's first (representative) read
            if difflib.SequenceMatcher(None, p, c["reads"][0][0]).ratio() >= 0.7:
                c["reads"].append((p, d["confidence"]))
                c["dets"].append(d)
                placed = True
                break
        if not placed:
            clusters.append({"reads": [(p, d["confidence"])], "dets": [d]})

    best: dict = {}
    for c in clusters:
        voted = _majority_vote(c["reads"])
        if not voted:
            continue
        plate = _postfix(voted[0])
        if not _looks_like_plate(plate):
            # Voting produced something invalid — keep the strongest raw read
            plate = max(c["reads"], key=lambda x: x[1])[0]
        conf = round(voted[1], 3)

        # Best bbox = the highest-confidence detection in this cluster
        bd = max(c["dets"], key=lambda x: x["confidence"])
        # Rewrite every frame's label so the burned-in overlay matches the table
        for det in c["dets"]:
            det["plate"] = plate

        norm = _norm(plate)
        if norm not in best or conf > best[norm]["confidence"]:
            best[norm] = {"plate": plate, "confidence": conf, "bbox": bd.get("bbox")}
    return best


def _scale_box(box, factor: float):
    return tuple(int(v * factor) for v in box)


def _attach_vehicle_boxes(frame: np.ndarray, detections: List[dict]) -> None:
    """Run the vehicle detector ONCE on a frame that already has a plate, and
    attach the surrounding vehicle's bounding box to each plate detection
    (key 'vehicle_bbox'). The annotator then draws a green/red box around the
    whole car. Detector runs only on the handful of plate frames, so the extra
    cost is small. Degrades silently to plate-only boxes if YOLO is unavailable.
    All coordinates stay in `frame` space — the caller scales them if needed.
    """
    try:
        det = get_detector()
        if not getattr(det, "available", False):
            return
        vehicles = det.detect(frame)
    except Exception:
        return
    if not vehicles:
        return

    for d in detections:
        pb = d.get("bbox")
        if pb:
            px = (pb[0] + pb[2]) / 2
            py = (pb[1] + pb[3]) / 2
            # Prefer the vehicle box that contains the plate centre …
            containing = [
                v for v in vehicles
                if v.bbox[0] <= px <= v.bbox[2] and v.bbox[1] <= py <= v.bbox[3]
            ]
            pool = containing or vehicles
            chosen = min(
                pool,
                key=lambda v: ((v.bbox[0] + v.bbox[2]) / 2 - px) ** 2
                            + ((v.bbox[1] + v.bbox[3]) / 2 - py) ** 2,
            )
        else:
            # No plate box — use the largest vehicle in frame
            chosen = max(
                vehicles,
                key=lambda v: (v.bbox[2] - v.bbox[0]) * (v.bbox[3] - v.bbox[1]),
            )
        d["vehicle_bbox"] = tuple(chosen.bbox)


def _build_track_json(
    video_path: str,
    fps: float,
    width: int,
    height: int,
    decision: str,
    total: int,
    job_id: str,
) -> None:
    """Sample the vehicle position ~10×/second across the whole clip and save it
    as JSON. The front-end overlays a box on the playing video and interpolates
    between these samples so it follows the vehicle. Runs in the background
    encode thread, so it never blocks the scan result.
    """
    try:
        det = get_detector()
        if not getattr(det, "available", False):
            return
    except Exception:
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return

    f       = fps or 25
    step    = max(1, int(round(f / 10)))    # ~10 samples per second
    limit   = min(total or _ANNOTATE_MAX_FRAMES, _ANNOTATE_MAX_FRAMES)
    boxes   = []
    idx     = 0
    while idx < limit:
        if not cap.grab():
            break
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            fh, fw = frame.shape[:2]
            sf = 1.0
            if fw > 960:
                sf = 960.0 / fw
                frame = cv2.resize(frame, (960, int(fh * sf)))
            try:
                vehicles = det.detect(frame)
            except Exception:
                vehicles = []
            if vehicles:
                v = max(vehicles, key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]))
                inv = 1.0 / sf
                x1, y1, x2, y2 = v.bbox
                boxes.append({
                    "t": round(idx / f, 3),
                    "x": int(x1 * inv), "y": int(y1 * inv),
                    "w": int((x2 - x1) * inv), "h": int((y2 - y1) * inv),
                })
        idx += 1
    cap.release()

    if not boxes:
        return
    payload = {
        "fps": f, "width": width, "height": height,
        "decision": decision, "boxes": boxes,
    }
    (UPLOADS / f"anpr_{job_id}_track.json").write_text(json.dumps(payload))


def _update_job(job_id: str, status: str, progress: float, error: Optional[str]):
    conn = db.get_conn()
    conn.execute(
        "UPDATE anpr_jobs SET status=?, progress=?, error=? WHERE job_id=?",
        (status, progress, error, job_id),
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Frame / video annotation helpers
# ---------------------------------------------------------------------------

def _annotate_frame(
    frame: np.ndarray,
    detections: List[dict],
    auth_map: dict,
) -> np.ndarray:
    """Draw plate bounding boxes with corner brackets and AUTHORIZED/DENIED labels."""
    out = frame.copy()
    fh, fw = out.shape[:2]
    font  = cv2.FONT_HERSHEY_SIMPLEX
    BLACK = (0, 0, 0)

    for d in detections:
        bbox = d.get("bbox")
        norm = _norm(d["plate"])
        auth = auth_map.get(norm)

        color = (50, 220, 90) if auth else (40, 40, 235)   # BGR: green / red
        label = d["plate"] + ("  AUTHORIZED" if auth else "  DENIED")

        # ---- Vehicle box: green if authorised, red if not ----
        vbox = d.get("vehicle_bbox")
        if vbox:
            vx1, vy1, vx2, vy2 = (
                max(0, vbox[0]), max(0, vbox[1]),
                min(fw - 1, vbox[2]), min(fh - 1, vbox[3]),
            )
            vw = vx2 - vx1
            # Thick rectangle around the whole vehicle (+ dark outline for contrast)
            cv2.rectangle(out, (vx1 - 2, vy1 - 2), (vx2 + 2, vy2 + 2), BLACK, 6)
            cv2.rectangle(out, (vx1, vy1), (vx2, vy2), color, 3)
            # Corner brackets (targeting-style)
            cl = max(14, min(vw // 6, 46))
            for cx, cy, sx, sy in [
                (vx1, vy1, 1, 1), (vx2, vy1, -1, 1),
                (vx1, vy2, 1, -1), (vx2, vy2, -1, -1),
            ]:
                cv2.line(out, (cx, cy), (cx + sx * cl, cy), color, 5)
                cv2.line(out, (cx, cy), (cx, cy + sy * cl), color, 5)
            # Label banner above the vehicle
            fs = max(0.55, min(1.0, vw / 420))
            (tw, th), bl = cv2.getTextSize(label, font, fs, 2)
            ly2 = vy1
            ly1 = max(0, vy1 - th - bl - 12)
            lx2 = min(fw, vx1 + tw + 16)
            cv2.rectangle(out, (vx1 - 1, ly1 - 1), (lx2 + 1, ly2 + 1), BLACK, -1)
            cv2.rectangle(out, (vx1, ly1), (lx2, ly2), color, -1)
            cv2.putText(out, label, (vx1 + 7, ly2 - bl - 3),
                        font, fs, (255, 255, 255), 2, cv2.LINE_AA)
            # Thin highlight on the plate itself (no extra label)
            if bbox:
                px1, py1, px2, py2 = (
                    max(0, bbox[0]), max(0, bbox[1]),
                    min(fw - 1, bbox[2]), min(fh - 1, bbox[3]),
                )
                cv2.rectangle(out, (px1, py1), (px2, py2), color, 2)
            continue

        if bbox:
            x1, y1, x2, y2 = (
                max(0, bbox[0]), max(0, bbox[1]),
                min(fw - 1, bbox[2]), min(fh - 1, bbox[3]),
            )
            bw = x2 - x1

            # Dark shadow outline — ensures visibility on any background
            cv2.rectangle(out, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), BLACK, 5)
            # Main plate rectangle
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)

            # Corner-bracket markers (L-shaped, CCTV style)
            cl = max(8, min(bw // 4, 22))   # corner arm length
            for cx, cy, sx, sy in [
                (x1, y1,  1,  1),   # top-left
                (x2, y1, -1,  1),   # top-right
                (x1, y2,  1, -1),   # bottom-left
                (x2, y2, -1, -1),   # bottom-right
            ]:
                cv2.line(out, (cx, cy), (cx + sx * cl, cy), color, 4)
                cv2.line(out, (cx, cy), (cx, cy + sy * cl), color, 4)

            # Label banner above the box
            fs = max(0.38, min(0.82, bw / 270))
            (tw, th), bl = cv2.getTextSize(label, font, fs, 2)
            ly2 = y1
            ly1 = max(0, y1 - th - bl - 10)
            lx2 = min(fw, x1 + tw + 14)
            # Shadow behind label background
            cv2.rectangle(out, (x1 - 1, ly1 - 1), (lx2 + 1, ly2 + 1), BLACK, -1)
            cv2.rectangle(out, (x1, ly1), (lx2, ly2), color, -1)
            cv2.putText(out, label, (x1 + 6, ly2 - bl - 2),
                        font, fs, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            # No bbox: banner at bottom of frame
            fs = 0.65
            (tw, th), bl = cv2.getTextSize(label, font, fs, 2)
            px, py = 12, fh - 20
            cv2.rectangle(out, (px - 5, py - th - 5), (px + tw + 5, py + bl + 2), BLACK, -1)
            cv2.rectangle(out, (px - 4, py - th - 4), (px + tw + 4, py + bl + 1), color, -1)
            cv2.putText(out, label, (px, py), font, fs, (255, 255, 255), 2, cv2.LINE_AA)

    return out


def _write_annotated_video(
    video_path: str,
    frame_detections: dict,   # frame_num → list[detection]
    auth_map: dict,
    out_path: str,
    total_frames: int,
    fps: float,
    width: int,
    height: int,
    job_id: str,
) -> bool:
    """Write MP4 with per-vehicle plate overlays that persist ~1.5 s each."""
    if not frame_detections:
        return False

    persist = max(1, int(fps * 1.5))

    # Build active_at[frame] = list of detections visible at that frame
    active_at: dict = {}
    for fnum, dets in frame_detections.items():
        for f in range(fnum, min(total_frames + 1, fnum + persist)):
            if f not in active_at:
                active_at[f] = []
            seen = {_norm(d["plate"]) for d in active_at[f]}
            for d in dets:
                if _norm(d["plate"]) not in seen:
                    active_at[f].append(d)
                    seen.add(_norm(d["plate"]))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        return False

    max_frames = min(total_frames, _ANNOTATE_MAX_FRAMES)
    frame_num = 0
    while frame_num < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        dets = active_at.get(frame_num)
        if dets:
            frame = _annotate_frame(frame, dets, auth_map)
        writer.write(frame)
        frame_num += 1
        if frame_num % 150 == 0:
            pct = 60 + (frame_num / max_frames) * 35
            _update_job(job_id, "processing", min(95, pct), None)

    cap.release()
    writer.release()
    return True


# ---------------------------------------------------------------------------
# Background job — runs in Starlette thread pool (sync function)
# ---------------------------------------------------------------------------

def _encode_video_background(
    video_path: str,
    frame_detections: dict,
    auth_map: dict,
    job_id: str,
    total: int,
    fps: float,
    width: int,
    height: int,
    decision: str = "denied",
):
    """After the scan result is already returned: (1) build the vehicle-tracking
    JSON the front-end uses to draw a moving box, and (2) encode the annotated
    MP4. Deletes the source video when done."""
    try:
        _build_track_json(video_path, fps, width, height, decision, total, job_id)
    except Exception:
        import traceback
        print(f"[ANPR track {job_id}] {traceback.format_exc()}")
    try:
        ann_path = UPLOADS / f"anpr_{job_id}_annotated.mp4"
        _write_annotated_video(
            video_path, frame_detections, auth_map,
            str(ann_path), total, fps, width, height, job_id,
        )
    except Exception:
        import traceback
        print(f"[ANPR encode {job_id}] {traceback.format_exc()}")
    finally:
        Path(video_path).unlink(missing_ok=True)


def _process_video(job_id: str, video_path: str):
    handed_off = False
    try:
        _update_job(job_id, "processing", 0, None)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            _update_job(job_id, "error", 0, "Cannot open video file")
            Path(video_path).unlink(missing_ok=True)
            return

        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        fps    = cap.get(cv2.CAP_PROP_FPS) or 25
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Sequential read: cap.grab() advances without full pixel decode (3-5× faster
        # than cap.set() seeking in compressed H.264/MP4 video).
        # Short clips get denser sampling so we don't miss the clearest plate frame.
        if total <= int(fps * 30):              # ≤ 30 s video
            step = max(1, int(fps * 0.5))       # every 0.5 s
        else:
            step = max(1, int(fps * 1.0))       # every 1.0 s for longer clips
        max_scan  = min(total, int(fps * 300))  # scan at most 5 minutes

        plates_best:       dict = {}
        frame_detections:  dict = {}
        plate_seen_count:  dict = {}   # norm → frames where it was detected
        all_dets:          List[dict] = []   # every read, for cross-frame voting
        frame_num      = 0
        no_new_streak  = 0

        while frame_num < max_scan:
            do_decode = (frame_num % step == 0)
            if do_decode:
                ret, frame = cap.read()
            else:
                ret = cap.grab()

            if not ret:
                break

            if do_decode:
                # Downscale wide frames before OCR — much faster on HD/4K clips.
                fh, fw = frame.shape[:2]
                if fw > _MAX_SCAN_W:
                    sf = _MAX_SCAN_W / fw
                    scan_frame = cv2.resize(frame, (_MAX_SCAN_W, int(fh * sf)))
                else:
                    sf = 1.0
                    scan_frame = frame

                detections = extract_plates_from_frame(scan_frame)
                improved = False
                if detections:
                    # Draw a box around the whole vehicle (green/red later) — detector
                    # runs only on this plate frame, so the speed cost is minimal.
                    _attach_vehicle_boxes(scan_frame, detections)
                    # Map all boxes back to full-resolution coordinates.
                    if sf != 1.0:
                        inv = 1.0 / sf
                        for d in detections:
                            if d.get("bbox"):
                                d["bbox"] = _scale_box(d["bbox"], inv)
                            if d.get("vehicle_bbox"):
                                d["vehicle_bbox"] = _scale_box(d["vehicle_bbox"], inv)

                    frame_detections[frame_num] = detections
                    all_dets.extend(detections)
                    for d in detections:
                        norm = _norm(d["plate"])
                        plate_seen_count[norm] = plate_seen_count.get(norm, 0) + 1
                        if norm not in plates_best or d["confidence"] > plates_best[norm]["confidence"] + 0.05:
                            plates_best[norm] = d
                            improved = True

                no_new_streak = 0 if improved else no_new_streak + 1

                # Instant stop: one very confident read (already char-corrected) is
                # enough — avoids scanning further frames when the plate is clear.
                if any(plates_best[n]["confidence"] >= 0.82 for n in plates_best):
                    break

                # Stop once a plate is confirmed in 2+ frames with decent confidence.
                # Cross-frame voting still corrects characters with 2 samples, and
                # stopping sooner keeps scanning fast.
                if any(
                    plate_seen_count.get(n, 0) >= 2 and plates_best[n]["confidence"] >= 0.65
                    for n in plates_best
                ):
                    break

                # Fallback stop: no improvement for 5 consecutive decoded frames
                if plates_best and no_new_streak >= 5:
                    break

                if frame_num % (step * 2) == 0:
                    _update_job(job_id, "processing",
                                min(88, frame_num / max_scan * 88), None)

            frame_num += 1

        cap.release()

        # Fallback: pre-filter rejected everything → try first 10 frames without filter
        if not plates_best:
            cap2 = cv2.VideoCapture(video_path)
            for _ in range(min(10, total)):
                ret, frame = cap2.read()
                if not ret:
                    break
                detections = extract_plates_from_frame(frame)
                if detections:
                    frame_detections[frame_num] = detections
                    all_dets.extend(detections)
                    for d in detections:
                        norm = _norm(d["plate"])
                        if norm not in plates_best or d["confidence"] > plates_best[norm]["confidence"]:
                            plates_best[norm] = d
                    if plates_best:
                        break
            cap2.release()

        # Accuracy pass: cross-frame majority vote over every read collected above.
        if all_dets:
            voted_best = _cluster_and_vote(all_dets)
            if voted_best:
                plates_best = voted_best

        # Cross-check against whitelist
        conn = db.get_conn()
        auth_rows = conn.execute(
            "SELECT plate, owner, vehicle_type FROM authorized_vehicles"
        ).fetchall()
        auth_map = {_norm(r["plate"]): dict(r) for r in auth_rows}

        # Persist results to DB immediately
        for norm, data in plates_best.items():
            auth     = auth_map.get(norm)
            decision = "GRANTED" if auth else "DENIED"
            conn.execute(
                "INSERT INTO anpr_log (plate, confidence, authorized, decision) VALUES (?,?,?,?)",
                (data["plate"], data["confidence"], 1 if auth else 0, decision),
            )
            conn.execute(
                "INSERT INTO anpr_plates (job_id, plate, confidence, authorized, owner, vehicle_type) "
                "VALUES (?,?,?,?,?,?)",
                (
                    job_id, data["plate"], data["confidence"],
                    1 if auth else 0,
                    auth["owner"]        if auth else "Unknown",
                    auth["vehicle_type"] if auth else "Unknown",
                ),
            )

        conn.commit()
        conn.close()

        # Mark completed NOW — results are in DB, frontend can show them immediately
        _update_job(job_id, "completed", 100, None)

        # Overall gate decision drives the tracking-box colour (green/red).
        decision = "granted" if any(auth_map.get(n) for n in plates_best) else "denied"

        # Background: build the vehicle-tracking JSON (+ annotated MP4). Always run
        # so the moving box shows even when the plate is unreadable / not authorised.
        handed_off = True
        threading.Thread(
            target=_encode_video_background,
            args=(video_path, frame_detections, auth_map,
                  job_id, total, fps, width, height, decision),
            daemon=True,
        ).start()

    except Exception as exc:
        import traceback
        print(f"[ANPR job {job_id}] {traceback.format_exc()}")
        _update_job(job_id, "error", 0, str(exc))
    finally:
        if not handed_off:
            Path(video_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/upload")
async def upload_video(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
):
    job_id     = str(uuid.uuid4())
    suffix     = Path(video.filename or "video.mp4").suffix or ".mp4"
    video_path = UPLOADS / f"anpr_{job_id}{suffix}"
    UPLOADS.mkdir(parents=True, exist_ok=True)

    # Stream the upload straight to disk in 1 MB chunks instead of buffering the
    # whole file in RAM (`await video.read()` allocates the entire video at once —
    # slow and memory-heavy for large phone clips). shutil.copyfileobj overlaps
    # read/write and keeps memory flat regardless of file size.
    import shutil
    def _save():
        with open(video_path, "wb") as out:
            shutil.copyfileobj(video.file, out, length=1024 * 1024)
    await asyncio.to_thread(_save)

    conn = db.get_conn()
    conn.execute(
        "INSERT INTO anpr_jobs (job_id, status, progress) VALUES (?,?,?)",
        (job_id, "pending", 0),
    )
    conn.commit()
    conn.close()

    background_tasks.add_task(_process_video, job_id, str(video_path))
    return {"job_id": job_id}


@router.get("/job/{job_id}/video")
def get_job_video(job_id: str):
    """Return the annotated output video for a completed job."""
    video_path = UPLOADS / f"anpr_{job_id}_annotated.mp4"
    if not video_path.exists():
        raise HTTPException(404, "No annotated video available for this job")
    return FileResponse(
        str(video_path),
        media_type="video/mp4",
        headers={"Accept-Ranges": "bytes"},
    )


@router.get("/job/{job_id}/track")
def get_job_track(job_id: str):
    """Vehicle-position samples for the moving detection box (front-end overlay)."""
    track_path = UPLOADS / f"anpr_{job_id}_track.json"
    if not track_path.exists():
        raise HTTPException(404, "No track available for this job")
    return FileResponse(str(track_path), media_type="application/json")


@router.get("/job/{job_id}/frame")
def get_job_frame(job_id: str):
    """Return a fallback annotated JPEG frame (kept for backwards compat)."""
    frame_path = UPLOADS / f"anpr_{job_id}_frame.jpg"
    if not frame_path.exists():
        raise HTTPException(404, "No annotated frame available for this job")
    return FileResponse(str(frame_path), media_type="image/jpeg")


@router.get("/job/{job_id}")
def get_job(job_id: str):
    conn = db.get_conn()
    job  = conn.execute("SELECT * FROM anpr_jobs WHERE job_id=?", (job_id,)).fetchone()
    if not job:
        conn.close()
        raise HTTPException(404, "Job not found")

    job = dict(job)
    # Return plates at any stage so the UI can show results as they arrive
    rows   = conn.execute(
        "SELECT * FROM anpr_plates WHERE job_id=?", (job_id,)
    ).fetchall()
    plates = [{**dict(r), "authorized": bool(r["authorized"])} for r in rows]
    conn.close()

    has_video = (UPLOADS / f"anpr_{job_id}_annotated.mp4").exists()
    return {**job, "plates": plates, "has_video": has_video}


@router.post("/scan-image")
async def scan_image(image: UploadFile = File(...)):
    """Fast single-image ANPR: decode a photo / screenshot, read any plate,
    cross-check the whitelist and return the decision immediately (no job/poll)."""
    content = await image.read()

    def _work():
        arr   = np.frombuffer(content, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Could not decode image — unsupported format?")

        fh, fw = frame.shape[:2]
        if fw > _MAX_SCAN_W:
            sf   = _MAX_SCAN_W / fw
            scan = cv2.resize(frame, (_MAX_SCAN_W, int(fh * sf)))
        else:
            sf, scan = 1.0, frame

        dets = extract_plates_from_frame(scan)
        if dets:
            _attach_vehicle_boxes(scan, dets)
            if sf != 1.0:
                inv = 1.0 / sf
                for d in dets:
                    if d.get("bbox"):
                        d["bbox"] = _scale_box(d["bbox"], inv)
                    if d.get("vehicle_bbox"):
                        d["vehicle_bbox"] = _scale_box(d["vehicle_bbox"], inv)

        # Keep the strongest read per normalised plate
        best: dict = {}
        for d in dets:
            n = _norm(d["plate"])
            if not n:
                continue
            if n not in best or d["confidence"] > best[n]["confidence"]:
                best[n] = d

        conn = db.get_conn()
        auth_rows = conn.execute(
            "SELECT plate, owner, vehicle_type FROM authorized_vehicles"
        ).fetchall()
        auth_map = {_norm(r["plate"]): dict(r) for r in auth_rows}

        plates = []
        for n, d in best.items():
            auth     = auth_map.get(n)
            decision = "GRANTED" if auth else "DENIED"
            conn.execute(
                "INSERT INTO anpr_log (plate, confidence, authorized, decision) VALUES (?,?,?,?)",
                (d["plate"], round(float(d["confidence"]), 4), 1 if auth else 0, decision),
            )
            plates.append({
                "plate":        d["plate"],
                "confidence":   round(float(d["confidence"]), 3),
                "authorized":   bool(auth),
                "owner":        auth["owner"]        if auth else "Unknown",
                "vehicle_type": auth["vehicle_type"] if auth else "Unknown",
            })
        conn.commit()
        conn.close()

        annotated = _annotate_frame(frame, list(best.values()), auth_map) if best else frame
        ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
        img_b64 = ("data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()
                   if ok else None)
        return {"plates": plates, "annotated_image": img_b64}

    try:
        return await asyncio.to_thread(_work)
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.get("/authorized")
def list_authorized():
    conn  = db.get_conn()
    rows  = conn.execute("SELECT plate, owner, vehicle_type FROM authorized_vehicles").fetchall()
    conn.close()
    return [dict(r) for r in rows]


@router.post("/authorized", status_code=201)
def add_authorized(body: AuthVehicle):
    norm = _norm(body.plate)
    conn = db.get_conn()
    existing = conn.execute(
        "SELECT 1 FROM authorized_vehicles WHERE plate=?", (norm,)
    ).fetchone()
    if existing:
        conn.close()
        raise HTTPException(409, f"Plate {norm} already in whitelist")
    conn.execute(
        "INSERT INTO authorized_vehicles (plate, owner, vehicle_type) VALUES (?,?,?)",
        (norm, body.owner, body.vehicle_type),
    )
    conn.commit()
    conn.close()
    return {"plate": norm, "owner": body.owner, "vehicle_type": body.vehicle_type}


@router.put("/authorized/{plate}")
def update_authorized(plate: str, body: AuthVehicleUpdate):
    """Update an authorised vehicle's owner / type, and optionally rename the plate.
    Lets the operator turn an 'Unknown' owner into a real name in place."""
    norm = _norm(plate)
    conn = db.get_conn()
    row = conn.execute(
        "SELECT plate, owner, vehicle_type FROM authorized_vehicles WHERE plate=?",
        (norm,),
    ).fetchone()
    if not row:
        conn.close()
        raise HTTPException(404, f"Plate {norm} not in whitelist")

    new_norm = _norm(body.new_plate) if body.new_plate else norm
    owner    = body.owner        if body.owner        is not None else row["owner"]
    vtype    = body.vehicle_type if body.vehicle_type is not None else row["vehicle_type"]

    if new_norm != norm:
        clash = conn.execute(
            "SELECT 1 FROM authorized_vehicles WHERE plate=?", (new_norm,)
        ).fetchone()
        if clash:
            conn.close()
            raise HTTPException(409, f"Plate {new_norm} already in whitelist")

    conn.execute(
        "UPDATE authorized_vehicles SET plate=?, owner=?, vehicle_type=? WHERE plate=?",
        (new_norm, owner, vtype, norm),
    )
    conn.commit()
    conn.close()
    return {"plate": new_norm, "owner": owner, "vehicle_type": vtype}


@router.delete("/authorized/{plate}", status_code=204)
def remove_authorized(plate: str):
    norm = _norm(plate)
    conn = db.get_conn()
    conn.execute("DELETE FROM authorized_vehicles WHERE plate=?", (norm,))
    conn.commit()
    conn.close()


@router.get("/log")
def access_log(limit: int = Query(30, ge=1, le=200)):
    conn = db.get_conn()
    rows = conn.execute(
        "SELECT timestamp, plate, confidence, authorized, decision "
        "FROM anpr_log ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return [
        {**dict(r), "authorized": bool(r["authorized"])} for r in rows
    ]
