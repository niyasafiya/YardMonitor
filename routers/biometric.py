"""
Biometric face-verification endpoints.
Register → store encoding; Verify → compare against DB.
"""
from __future__ import annotations

import asyncio
import tempfile
import uuid
from pathlib import Path
from typing import Optional

import cv2
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel

import db
from services import bio_service

router = APIRouter()

# Reuse the same temp dir as the ANPR module
_UPLOADS = Path(tempfile.gettempdir()) / "sentinel_ai"

# Lazy person detector (YOLOv4-tiny via OpenCV DNN — detects PERSON, unlike the
# vehicle-only Detector in services/yolo_service.py)
_people_detector = None


def _get_people_detector():
    global _people_detector
    if _people_detector is None:
        from detector import YOLODetector
        _people_detector = YOLODetector(conf_thresh=0.35)
    return _people_detector


def _scan_people_video(video_path: str) -> dict:
    """Run person detection across an uploaded clip. Returns a verdict plus a
    per-time box of the main person so the front-end can draw a moving box.

    decision: 'granted' (one person, access ok) | 'tailgating' (2+ in a frame)
              | 'none' (no person found) | 'unavailable' (detector missing)
    """
    try:
        det = _get_people_detector()
    except Exception:
        return {"decision": "unavailable", "people_max": 0,
                "frames_with_person": 0, "boxes": []}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise HTTPException(400, "Cannot open video file")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    step   = max(1, int(round(fps / 8)))     # ~8 samples/sec
    limit  = min(total or 6000, 6000)

    boxes, people_max, frames_with_person, idx = [], 0, 0, 0
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
                dets = det.detect(frame)
            except Exception:
                dets = []
            persons = [d for d in dets if getattr(d, "label", "") == "PERSON"]
            if persons:
                frames_with_person += 1
                people_max = max(people_max, len(persons))
                p = max(persons, key=lambda d: (d.x2 - d.x1) * (d.y2 - d.y1))
                inv = 1.0 / sf
                boxes.append({
                    "t": round(idx / fps, 3),
                    "x": int(p.x1 * inv), "y": int(p.y1 * inv),
                    "w": int((p.x2 - p.x1) * inv), "h": int((p.y2 - p.y1) * inv),
                })
        idx += 1
    cap.release()

    if people_max >= 2:
        decision = "tailgating"
    elif frames_with_person > 0:
        decision = "granted"
    else:
        decision = "none"

    return {
        "decision": decision, "people_max": people_max,
        "frames_with_person": frames_with_person,
        "fps": fps, "width": width, "height": height, "boxes": boxes,
    }


class UpdatePerson(BaseModel):
    name:            Optional[str] = None
    department:      Optional[str] = None
    clearance_level: Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _list_persons() -> list[dict]:
    conn  = db.get_conn()
    rows  = conn.execute(
        "SELECT employee_id, name, department, clearance_level, photo_path "
        "FROM persons ORDER BY name"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/register", status_code=201)
async def register_person(
    photo:           UploadFile  = File(...),
    name:            str         = Form(...),
    employee_id:     str         = Form(...),
    department:      str         = Form("General"),
    clearance_level: str         = Form("L1"),
):
    image_bytes = await photo.read()

    # Save photo
    photo_path = bio_service.save_photo(employee_id, image_bytes)

    # Compute & store face embedding
    embedding     = bio_service.compute_embedding(image_bytes)
    face_detected = bool(embedding and embedding.get("face", True))
    if embedding:
        bio_service.save_encoding(employee_id, embedding)

    # Upsert person record
    conn = db.get_conn()
    conn.execute(
        """
        INSERT INTO persons (employee_id, name, department, clearance_level, photo_path)
        VALUES (?,?,?,?,?)
        ON CONFLICT(employee_id) DO UPDATE SET
            name=excluded.name, department=excluded.department,
            clearance_level=excluded.clearance_level, photo_path=excluded.photo_path
        """,
        (employee_id, name, department, clearance_level, photo_path),
    )
    conn.commit()
    conn.close()

    return {
        "success":      True,
        "face_detected": face_detected,
        "message":      (
            f"{name} registered with face embedding."
            if face_detected
            else f"{name} registered (no face detected — use a clearer photo)."
        ),
    }


@router.post("/verify")
async def verify_face(photo: UploadFile = File(...), log: bool = Query(True)):
    """Verify a face. `log=false` skips the access-log write — used by the live
    webcam loop so it doesn't flood the log on every sampled frame."""
    image_bytes = await photo.read()
    persons     = _list_persons()
    result      = bio_service.verify_face(image_bytes, persons)

    if log:
        decision = "GRANTED" if result["matched"] else "DENIED"
        person   = result.get("person") or {}
        conn = db.get_conn()
        conn.execute(
            "INSERT INTO bio_log (person_name, confidence, decision) VALUES (?,?,?)",
            (person.get("name", "Unknown"), result["confidence"], decision),
        )
        conn.commit()
        conn.close()

    return result


@router.post("/face-video")
async def face_video(video: UploadFile = File(...)):
    """Gate face / tailgating demo: detect people in an uploaded clip and decide
    grant vs. tailgating alert. Returns a moving person-box track for the UI."""
    _UPLOADS.mkdir(parents=True, exist_ok=True)
    suffix = Path(video.filename or "v.mp4").suffix or ".mp4"
    tmp    = _UPLOADS / f"face_{uuid.uuid4().hex}{suffix}"
    tmp.write_bytes(await video.read())
    try:
        result = await asyncio.to_thread(_scan_people_video, str(tmp))
    finally:
        tmp.unlink(missing_ok=True)

    # Log to bio_log so it shows in the overview event feed
    dec_map = {"granted": "GRANTED", "tailgating": "DENIED",
               "none": "DENIED", "unavailable": "DENIED"}
    name_map = {"granted": "Authorized person", "tailgating": "Tailgating — 2+ people",
                "none": "No face detected", "unavailable": "Detector unavailable"}
    try:
        conn = db.get_conn()
        conn.execute(
            "INSERT INTO bio_log (person_name, confidence, decision) VALUES (?,?,?)",
            (name_map.get(result["decision"], "Face check"),
             1.0 if result["decision"] == "granted" else 0.0,
             dec_map.get(result["decision"], "DENIED")),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass

    return result


def _recognize_face_video(video_path: str, persons: list[dict]) -> dict:
    """Sample frames from a clip and run face recognition against the registered
    people. Returns the best match found (so a single good frame is enough)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise HTTPException(400, "Cannot open video file")

    fps   = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    step  = max(1, int(round(fps * 0.6)))      # ~1.6 frames/sec
    limit = min(total or 1500, 1500)           # scan ≤ ~60 s
    MAX_CHECKS = 16                            # a few more frames → better odds of a clear face

    best = {"matched": False, "confidence": 0.0, "person": None, "engine": "none"}
    idx = 0
    frames_checked = 0
    while idx < limit and frames_checked < MAX_CHECKS:
        if not cap.grab():
            break
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            frames_checked += 1
            fh, fw = frame.shape[:2]
            if fw > 800:
                frame = cv2.resize(frame, (800, int(fh * 800 / fw)))
            ok2, buf = cv2.imencode(".jpg", frame)
            if ok2:
                r = bio_service.verify_face(buf.tobytes(), persons)
                if r["confidence"] > best["confidence"]:
                    best = r
                # Confident match — stop early. ArcFace genuine cosine sits well
                # above its 0.40 match threshold, so 0.55 is a safe "sure" bar.
                if r["matched"] and r["confidence"] >= 0.55:
                    break
        idx += 1
    cap.release()
    best["frames_checked"] = frames_checked
    return best


@router.post("/verify-video")
async def verify_video(video: UploadFile = File(...)):
    """Face recognition over an uploaded clip. Grants access if a registered
    person is recognised in any sampled frame."""
    _UPLOADS.mkdir(parents=True, exist_ok=True)
    suffix = Path(video.filename or "v.mp4").suffix or ".mp4"
    tmp    = _UPLOADS / f"facever_{uuid.uuid4().hex}{suffix}"
    tmp.write_bytes(await video.read())

    persons = _list_persons()
    try:
        result = await asyncio.to_thread(_recognize_face_video, str(tmp), persons)
    finally:
        tmp.unlink(missing_ok=True)

    decision = "GRANTED" if result["matched"] else "DENIED"
    person   = result.get("person") or {}
    try:
        conn = db.get_conn()
        conn.execute(
            "INSERT INTO bio_log (person_name, confidence, decision) VALUES (?,?,?)",
            (person.get("name", "Unknown"), result["confidence"], decision),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass

    return result


def _collect_gallery(video_path: str):
    """Sample many frames across the clip and build a diverse GALLERY of face
    embeddings — different angles / expressions / with & without glasses. This is
    what makes verification robust to appearance changes. Returns
    (photo_jpeg_bytes, gallery) where gallery = {"engine", "vecs": [...]}."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise HTTPException(400, "Cannot open video file")

    fps   = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    step  = max(1, int(round(fps * 0.3)))      # ~3 frames/sec
    limit = min(total or 3000, 3000)

    photo   = None
    gallery: dict = {}
    idx = 0
    while idx < limit:
        if not cap.grab():
            break
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            fh, fw = frame.shape[:2]
            if fw > 960:
                frame = cv2.resize(frame, (960, int(fh * 960 / fw)))
            ok2, buf = cv2.imencode(".jpg", frame)
            if ok2:
                jpg = buf.tobytes()
                if photo is None:
                    photo = jpg                # keep first frame as a fallback photo
                emb = bio_service.compute_embedding(jpg)
                if emb and emb.get("vec"):
                    if not gallery.get("vecs"):
                        photo = jpg            # prefer a real face frame for the photo
                    bio_service.add_to_gallery(gallery, emb)
                    # Also add a horizontally-mirrored view → better recognition when
                    # the face is turned to the other side than in enrolment.
                    ok3, mbuf = cv2.imencode(".jpg", cv2.flip(frame, 1))
                    if ok3:
                        memb = bio_service.compute_embedding(mbuf.tobytes())
                        if memb and memb.get("vec"):
                            bio_service.add_to_gallery(gallery, memb)
                    # …and a contrast/sharpness-enhanced view → matches unclear or
                    # low-light query frames later.
                    enh = bio_service.enhance_bytes(jpg)
                    if enh:
                        eemb = bio_service.compute_embedding(enh)
                        if eemb and eemb.get("vec"):
                            bio_service.add_to_gallery(gallery, eemb)
                    if len(gallery.get("vecs", [])) >= bio_service._MAX_GALLERY:
                        break
        idx += 1
    cap.release()
    return photo, gallery


@router.post("/register-video", status_code=201)
async def register_person_video(
    video:           UploadFile = File(...),
    name:            str        = Form(...),
    employee_id:     str        = Form(...),
    department:      str        = Form("General"),
    clearance_level: str        = Form("L1"),
):
    """Register a person from a video clip: build a gallery of face embeddings
    across the clip (robust to glasses / angle / lighting) and store the best photo."""
    _UPLOADS.mkdir(parents=True, exist_ok=True)
    suffix = Path(video.filename or "v.mp4").suffix or ".mp4"
    tmp    = _UPLOADS / f"reg_{uuid.uuid4().hex}{suffix}"
    tmp.write_bytes(await video.read())
    try:
        frame_bytes, gallery = await asyncio.to_thread(_collect_gallery, str(tmp))
    finally:
        tmp.unlink(missing_ok=True)

    n_vecs        = len(gallery.get("vecs", []))
    face_detected = n_vecs > 0
    photo_path    = bio_service.save_photo(employee_id, frame_bytes) if frame_bytes else None
    if face_detected:
        bio_service.save_encoding(employee_id, gallery)

    conn = db.get_conn()
    conn.execute(
        """
        INSERT INTO persons (employee_id, name, department, clearance_level, photo_path)
        VALUES (?,?,?,?,?)
        ON CONFLICT(employee_id) DO UPDATE SET
            name=excluded.name, department=excluded.department,
            clearance_level=excluded.clearance_level, photo_path=excluded.photo_path
        """,
        (employee_id, name, department, clearance_level, photo_path),
    )
    conn.commit()
    conn.close()

    return {
        "success": True,
        "face_detected": face_detected,
        "samples": n_vecs,
        "message": (
            f"{name} registered from video — captured {n_vecs} face sample"
            f"{'' if n_vecs == 1 else 's'} for robust recognition."
            if face_detected
            else f"{name} registered, but no clear face was found — try a clearer clip."
        ),
    }


@router.get("/persons")
def list_persons():
    rows = _list_persons()
    return [
        {
            "employee_id":    r["employee_id"],
            "name":           r["name"],
            "department":     r["department"],
            "clearance_level": r["clearance_level"],
            "has_photo":      r["photo_path"] is not None,
        }
        for r in rows
    ]


@router.patch("/persons/{employee_id}")
def update_person(employee_id: str, body: UpdatePerson):
    conn = db.get_conn()
    row = conn.execute("SELECT employee_id FROM persons WHERE employee_id=?", (employee_id,)).fetchone()
    if not row:
        conn.close()
        raise HTTPException(404, "Person not found")

    updates: dict = {}
    if body.name is not None:            updates["name"]            = body.name
    if body.department is not None:      updates["department"]      = body.department
    if body.clearance_level is not None: updates["clearance_level"] = body.clearance_level

    if updates:
        set_clause = ", ".join(f"{k}=?" for k in updates)
        conn.execute(
            f"UPDATE persons SET {set_clause} WHERE employee_id=?",
            (*updates.values(), employee_id),
        )
        conn.commit()
    conn.close()
    return {"employee_id": employee_id, **updates}


@router.delete("/persons/{employee_id}", status_code=204)
def delete_person(employee_id: str):
    conn = db.get_conn()
    conn.execute("DELETE FROM persons WHERE employee_id=?", (employee_id,))
    conn.commit()
    conn.close()
    bio_service.delete_encoding(employee_id)


@router.post("/log", status_code=201)
def add_log_entry(
    person_name: str   = Form(""),
    confidence:  float = Form(0.0),
    decision:    str   = Form("GRANTED"),
):
    conn = db.get_conn()
    conn.execute(
        "INSERT INTO bio_log (person_name, confidence, decision) VALUES (?,?,?)",
        (person_name, round(float(confidence), 4), decision.upper()),
    )
    conn.commit()
    conn.close()
    return {"ok": True}


@router.get("/log")
def bio_log(limit: int = Query(30, ge=1, le=200)):
    conn  = db.get_conn()
    rows  = conn.execute(
        "SELECT timestamp, person_name, confidence, decision "
        "FROM bio_log ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
