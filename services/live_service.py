"""
Live vehicle + number-plate analytics over an MJPEG/RTSP source.

Designed for a phone used as an IP camera (e.g. the "IP Webcam" Android app,
which exposes  http://<phone-ip>:8080/video ) but works with any RTSP/MJPEG URL
or a local webcam index.

Performance model (why this stays smooth on CPU):
  • YOLOv8 + ByteTrack runs on EVERY streamed frame → a stable box + ID follows
    every vehicle in real time.
  • OCR (the expensive part) runs on a SEPARATE worker thread, throttled to a
    couple of the largest vehicles every few frames. The plate text is cached
    against the ByteTrack id and drawn on that vehicle's box, so each car gets
    its number as it comes into clear view without ever stalling the video.

The endpoint (routers/vehicles.py::live) simply wraps `live_mjpeg` in a
StreamingResponse with  multipart/x-mixed-replace  so an <img> tag shows it.
"""
from __future__ import annotations

import asyncio
import os
import queue
import threading
import time
from typing import Dict, Optional

import cv2
import numpy as np

import db
from services.anpr_service import read_plate_crop
from services.yolo_service import find_plate_contour, get_detector, get_plate_detector

# ── Tunables ────────────────────────────────────────────────────────────────
_STREAM_W       = 720      # downscale incoming frames to this width before detect/draw
_DET_IMGSZ      = 480      # YOLO inference size (raised 416→480: better recall on small /
                           # distant / blurred vehicles, without the frame-rate hit of 512 —
                           # and keeping the rate high matters most for FAST movers, which
                           # cross the scene in few frames)
_DET_CONF       = 0.30     # detection confidence (lowered from the 0.45 default): a
                           # motion-blurred fast car scores low, so a high bar drops it
_TRACKER_CFG    = "models/bytetrack_fast.yaml"  # keeps fast movers' ids stable
_OCR_EVERY      = 3        # consider OCR every N processed frames (was 4 — faster cars
                           # spend fewer frames on screen, so read more often)
_OCR_PER_TICK   = 3        # at most this many vehicles queued for OCR per tick
_MIN_BOX_W      = 60       # skip OCR on vehicles smaller than this (plate unreadable)
_LOCK_CONF      = 0.75     # stop re-reading a plate once we're this confident
_FAST_SPEED     = 14.0     # box-centre px/frame above which a track counts as "fast"
_PRUNE_AFTER    = 300      # forget a track not seen for this many frames (~free memory)
_JPEG_Q         = 78

# ── Adaptive detection stride (CPU-only anti-lag) ────────────────────────────
# YOLO runs on every frame by default. On a machine that can't keep up, that's
# the main source of lag. When the measured display FPS drops below _FPS_SLOW we
# run detection on every _DET_STRIDE_MAX-th frame instead and REUSE the previous
# boxes on the frames in between — the video still shows every frame, only the
# boxes refresh at half rate. Throughput ~doubles. Hysteresis (_FPS_OK) prevents
# flip-flopping. When the machine is fast (FPS ≥ _FPS_OK) stride stays 1, so on a
# capable host behaviour is IDENTICAL to before — the optimisation only activates
# under exactly the load that causes lag.
_DET_STRIDE_MAX = 2
_FPS_SLOW       = 9.0      # below this measured fps → skip-detect (stride up)
_FPS_OK         = 15.0     # above this → back to per-frame detection (stride down)

_COLORS = {                # BGR
    "known_auth":   (104, 210, 118),  # green  — plate read + whitelisted
    "known_deny":   (72,  86, 240),   # red    — plate read, not whitelisted
    "vehicle":      (245, 191,  66),  # amber/cyan — tracked, plate not read yet
}
_INK       = (235, 238, 242)          # near-white label text
_CHIP_BG   = (20, 22, 26)             # dark chip backing (BGR)


def _norm(plate: str) -> str:
    return plate.upper().replace(" ", "").replace("-", "")


def _load_auth_map() -> Dict[str, dict]:
    """{normalised_plate: {owner, vehicle_type}} from the whitelist."""
    try:
        conn = db.get_conn()
        rows = conn.execute(
            "SELECT plate, owner, vehicle_type FROM authorized_vehicles"
        ).fetchall()
        conn.close()
        return {_norm(r["plate"]): dict(r) for r in rows}
    except Exception:
        return {}


# ── OCR worker (runs off the streaming thread) ──────────────────────────────
def _ocr_worker(q: "queue.Queue", results: dict, auth_map: dict,
                stop_evt: threading.Event) -> None:
    """Consumes the sharpest VEHICLE crop per track, localises the plate inside it,
    then OCRs. Plate localisation (a YOLO pass) and OCR both run here, off the
    streaming thread, so heavy work never stalls the live video — critical for
    keeping up with fast movers."""
    while not stop_evt.is_set():
        try:
            tid, veh_crop = q.get(timeout=0.4)
        except queue.Empty:
            continue
        if veh_crop is None or veh_crop.size == 0:
            continue

        # Localise the plate inside the vehicle crop. Trained detector first, with
        # the contour heuristic as fallback (matches the previous behaviour, but now
        # off the streaming thread).
        box = None
        plate_det = get_plate_detector()
        if plate_det is not None:
            try:
                box = plate_det.detect_best(veh_crop)
            except Exception:
                box = None
        if box is None:
            box = find_plate_contour(veh_crop)
        px1, py1, px2, py2 = box
        px1, py1 = max(0, px1), max(0, py1)
        plate_crop = veh_crop[py1:py2, px1:px2]
        if plate_crop.size == 0:
            continue

        try:
            res = read_plate_crop(plate_crop)
        except Exception:
            res = None
        if not res:
            continue
        conf = float(res["confidence"])
        prev = results.get(tid)
        if prev is not None and prev["confidence"] >= conf:
            continue
        norm = _norm(res["plate"])
        auth = auth_map.get(norm)
        results[tid] = {
            "plate":      res["plate"],
            "confidence": conf,
            "authorized": auth is not None,
            "owner":      auth["owner"] if auth else "Unknown",
        }


# ── Source open helper ──────────────────────────────────────────────────────
def _configure(cap: cv2.VideoCapture) -> cv2.VideoCapture:
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # always show the latest frame (low latency)
    except Exception:
        pass
    return cap


def _open(src: str) -> cv2.VideoCapture:
    """Open a source, which may be:
      • a webcam / USB-virtual-camera index  ("0", "1", …)  — iVCam / EpocCam / DroidCam-USB
      • an rtsp:// URL                        — IP Camera Lite (iOS), CCTV
      • an http MJPEG URL                     — DroidCam / IP Webcam
    Prefers TCP for RTSP; for URLs, falls back to OpenCV's default backend when
    FFMPEG can't open the stream (some phone MJPEG servers need this)."""
    # TCP transport + a 5 s connect/read timeout so a wrong URL fails fast
    # (default FFMPEG behaviour hangs ~30 s on an unreachable host).
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;5000000"
    s = src.strip()
    if s.isdigit():
        idx = int(s)
        # DirectShow is far more reliable than the default MSMF backend for
        # USB / virtual webcams like iVCam / EpocCam / DroidCam on Windows.
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(idx)   # fallback to default backend
        return _configure(cap)

    cap = cv2.VideoCapture(s, cv2.CAP_FFMPEG)
    # Only retry with the default backend for HTTP MJPEG — for rtsp:// the default
    # backend can block ~30 s on an unreachable host, so we let FFMPEG own RTSP.
    if not cap.isOpened() and s.lower().startswith("http"):
        cap.release()
        cap = cv2.VideoCapture(s)   # recovers some phone MJPEG servers
    return _configure(cap)


# ── Drawing ─────────────────────────────────────────────────────────────────
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _alpha_rect(img, x1, y1, x2, y2, color, alpha):
    """Blend a solid `color` at `alpha` over a rectangular region (in place)."""
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return
    roi  = img[y1:y2, x1:x2]
    tint = np.empty_like(roi)
    tint[:] = color
    cv2.addWeighted(tint, alpha, roi, 1.0 - alpha, 0, roi)


def _corner_box(img, x1, y1, x2, y2, color, thick=2):
    """Modern ANPR/analytics box: a soft translucent fill, a hairline full frame,
    and bright rounded corner brackets that read cleanly at any scale."""
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return
    # bracket length scales with the box but never dominates a small one
    L = int(max(14, min(w, h) * 0.24))
    L = max(6, min(L, w // 2, h // 2))

    # subtle interior tint so the tracked vehicle "pops" without hiding it
    _alpha_rect(img, x1, y1, x2, y2, color, 0.10)
    # hairline full frame for context (drawn faint via a blended pass)
    frame_ov = img.copy()
    cv2.rectangle(frame_ov, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
    cv2.addWeighted(frame_ov, 0.45, img, 0.55, 0, img)

    # bright L-shaped corner brackets with a small rounded joint
    for cx, cy, dx, dy in ((x1, y1, 1, 1), (x2, y1, -1, 1),
                           (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(img, (cx, cy), (cx + dx * L, cy), color, thick, cv2.LINE_AA)
        cv2.line(img, (cx, cy), (cx, cy + dy * L), color, thick, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), max(1, thick - 1), color, -1, cv2.LINE_AA)


def _chip(img, x, y, text, color, fs=0.5):
    """A professional label pill: dark translucent backing, a colored accent bar,
    and crisp light text. Anchored with its bottom-left at (x, y)."""
    (tw, th), bl = cv2.getTextSize(text, _FONT, fs, 1)
    pad_x, pad_y = 8, 5
    accent = 4
    box_w = accent + pad_x * 2 + tw
    box_h = th + bl + pad_y * 2

    x1 = x
    y1 = y - box_h
    fw, fh = img.shape[1], img.shape[0]
    if x1 + box_w > fw:               # keep on-screen horizontally
        x1 = fw - box_w
    x1 = max(0, x1)
    y1 = max(0, min(y1, fh - box_h))
    x2, y2 = x1 + box_w, y1 + box_h

    # dark translucent backing → text stays legible over any scene
    _alpha_rect(img, x1, y1, x2, y2, _CHIP_BG, 0.78)
    # class-colored accent bar on the left edge
    cv2.rectangle(img, (x1, y1), (x1 + accent, y2), color, -1, cv2.LINE_AA)
    cv2.putText(img, text, (x1 + accent + pad_x, y2 - pad_y - bl + 1),
                _FONT, fs, _INK, 1, cv2.LINE_AA)


def _plate_zone_sharpness(crop: np.ndarray) -> float:
    """Laplacian variance of the plate zone (bottom ~55 % of a vehicle crop).
    Higher = sharper. Used to keep the least motion-blurred crop per track so OCR
    reads a crisp plate off a fast car instead of a smeared one."""
    h = crop.shape[0]
    zone = crop[int(h * 0.45):, :]
    if zone.size == 0:
        zone = crop
    gray = cv2.cvtColor(zone, cv2.COLOR_BGR2GRAY) if zone.ndim == 3 else zone
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _draw(frame: np.ndarray, dets, results: dict, meta: dict, fps: float) -> None:
    fast_n = 0
    for d in dets:
        x1, y1, x2, y2 = d.bbox
        m = meta.get(d.track_id) if d.track_id is not None else None
        is_fast = bool(m and m["speed"] >= _FAST_SPEED)
        if is_fast:
            fast_n += 1
        r = results.get(d.track_id) if d.track_id is not None else None
        if r:
            color = _COLORS["known_auth"] if r["authorized"] else _COLORS["known_deny"]
            label = f"{r['plate']}  {'AUTH' if r['authorized'] else 'DENY'}"
        else:
            color = _COLORS["vehicle"]
            tid   = f"#{d.track_id}" if d.track_id is not None else ""
            label = f"{d.cls_name.capitalize()} {tid}".strip()
            if is_fast:
                label += "  » FAST"
        _corner_box(frame, x1, y1, x2, y2, color, thick=2)
        # place the label above the box, or just inside the top if there's no room
        chip_y = y1 - 4 if y1 > 22 else y1 + 22
        _chip(frame, x1, chip_y, label, color,
              fs=0.46 if r is None else 0.52)

    # Translucent HUD strip: vehicle count (+ fast movers) + fps, with a status dot
    fast_txt = f"  ({fast_n} FAST)" if fast_n else ""
    hud = f"VEHICLES {len(dets)}{fast_txt}     {fps:4.1f} FPS"
    (tw, _), _ = cv2.getTextSize(hud, _FONT, 0.5, 1)
    bw = tw + 42
    _alpha_rect(frame, 0, 0, bw, 30, _CHIP_BG, 0.72)
    cv2.rectangle(frame, (0, 30), (bw, 31), (245, 191, 66), -1, cv2.LINE_AA)  # accent underline
    cv2.circle(frame, (16, 15), 4, (120, 230, 180), -1, cv2.LINE_AA)          # live dot
    cv2.putText(frame, hud, (30, 20), _FONT, 0.5, _INK, 1, cv2.LINE_AA)


def _process_frame(detector, frame: np.ndarray, frame_idx: int,
                   results: dict, meta: dict, q: "queue.Queue", fps: float,
                   run_detect: bool = True, prev_dets=None):
    """Process one frame and return (jpeg_bytes, detections).

    When `run_detect` is False we skip the (expensive) YOLO pass and reuse
    `prev_dets` — the video still shows this frame, only the boxes are one frame
    stale. The per-track velocity / sharpest-crop / OCR-queue work is likewise
    skipped on those frames because it's only meaningful on a fresh detection.
    """
    h, w = frame.shape[:2]
    if w > _STREAM_W:
        s = _STREAM_W / w
        frame = cv2.resize(frame, (_STREAM_W, int(h * s)))
    fw = frame.shape[1]

    if run_detect and detector.available:
        dets = detector.detect(frame, track=True, imgsz=_DET_IMGSZ,
                               conf=_DET_CONF, tracker=_TRACKER_CFG)
    else:
        # Reuse last detection (skip-detect frame) — keeps the stream smooth
        # under load without paying for YOLO on every frame.
        dets = prev_dets or []

    if not run_detect:
        # No fresh boxes → just draw the reused ones and emit. Velocity, crop
        # selection and OCR queueing all need a real detection, so they wait for
        # the next detect frame.
        _draw(frame, dets, results, meta, fps)
        ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_Q])
        return (jpg.tobytes() if ok else b""), dets

    # ── Per-track update: velocity (for prioritising / labelling fast movers) and
    #    the SHARPEST plate crop seen so far. Reading OCR off the crispest frame is
    #    what makes a fast car's plate legible — the instantaneous frame is usually
    #    motion-blurred. ─────────────────────────────────────────────────────────
    for d in dets:
        tid = d.track_id
        if tid is None:
            continue
        cx, cy = d.center
        m = meta.get(tid)
        if m is None:
            m = meta[tid] = {"cx": cx, "cy": cy, "speed": 0.0,
                             "best_crop": None, "best_sharp": -1.0,
                             "last_seen": frame_idx}
        else:
            disp = float(np.hypot(cx - m["cx"], cy - m["cy"]))
            m["speed"] = 0.6 * m["speed"] + 0.4 * disp   # smoothed px/frame
            m["cx"], m["cy"], m["last_seen"] = cx, cy, frame_idx

        r = results.get(tid)
        if (r is None or r["confidence"] < _LOCK_CONF) and d.width >= _MIN_BOX_W:
            crop = detector.get_vehicle_crop(frame, d)
            if crop.size:
                sharp = _plate_zone_sharpness(crop)
                if sharp > m["best_sharp"]:
                    # copy: _draw() will paint over `frame` after this
                    m["best_sharp"] = sharp
                    m["best_crop"]  = crop.copy()

    # ── Throttled OCR: queue the most URGENT vehicles first — those about to leave
    #    the frame, fast movers, then large/clear ones — so a car speeding out of
    #    view still gets read before it's gone. Localisation + OCR run in the worker.
    if dets and frame_idx % _OCR_EVERY == 0:
        cands = []
        for d in dets:
            tid = d.track_id
            if tid is None:
                continue
            r = results.get(tid)
            if r and r["confidence"] >= _LOCK_CONF:
                continue
            m = meta.get(tid)
            if not m or m["best_crop"] is None:
                continue
            edge     = min(m["cx"], fw - m["cx"]) / max(fw, 1)   # 0 at L/R edge … 0.5 centre
            exit_urg = 1.0 - min(edge / 0.25, 1.0)               # 1 hugging an edge, 0 well inside
            speed_urg = min(m["speed"] / _FAST_SPEED, 1.0)
            size_urg  = min(d.width / 300.0, 1.0)
            score = 2.0 * exit_urg + 1.3 * speed_urg + 0.6 * size_urg
            cands.append((score, tid, m))

        cands.sort(key=lambda c: c[0], reverse=True)
        enq = 0
        for _score, tid, m in cands:
            try:
                q.put_nowait((tid, m["best_crop"]))
            except queue.Full:
                break
            m["best_sharp"] = -1.0   # reset window → collect a fresh sharpest crop next
            m["best_crop"]  = None
            enq += 1
            if enq >= _OCR_PER_TICK:
                break

    _draw(frame, dets, results, meta, fps)
    ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_Q])
    return (jpg.tobytes() if ok else b""), dets


def _status_jpg(line1: str, line2: str = "") -> bytes:
    frame = np.full((360, 640, 3), (14, 18, 24), np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for txt, yp, sc, col in [
        (line1, 168, 0.7, (150, 190, 215)),
        (line2, 200, 0.42, (90, 130, 160)),
    ]:
        if not txt:
            continue
        tw = cv2.getTextSize(txt, font, sc, 1)[0][0]
        cv2.putText(frame, txt, ((640 - tw) // 2, yp), font, sc, col, 1, cv2.LINE_AA)
    ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return jpg.tobytes()


# ── Capture thread: always hold only the NEWEST frame ───────────────────────
class _Grabber:
    """Continuously reads the source in its own thread and keeps only the latest
    frame. Because we never queue frames, the processor always works on the most
    recent image — so detection running slower than the camera can NEVER build up
    latency (the video stays live instead of lagging further and further behind).
    Auto-reconnects if the source drops."""

    def __init__(self, src: str):
        self.src      = src
        self._lock    = threading.Lock()
        self._frame   = None
        self._seq     = 0
        self._status  = "connecting"     # connecting | live | offline
        self._stop    = threading.Event()
        self._thread  = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()

    def latest(self):
        with self._lock:
            return self._seq, self._frame, self._status

    def _run(self):
        cap = None
        fail = 0
        while not self._stop.is_set():
            if cap is None or not cap.isOpened():
                if cap is not None:
                    cap.release()
                with self._lock:
                    self._status = "connecting"
                cap = _open(self.src)
                if not cap.isOpened():
                    fail += 1
                    with self._lock:
                        self._status = "offline"
                    time.sleep(min(float(fail), 5.0))
                    continue
                fail = 0
            ok, frame = cap.read()
            if not ok or frame is None:
                cap.release()
                cap = None
                time.sleep(0.2)
                continue
            with self._lock:
                self._frame  = frame
                self._seq   += 1
                self._status = "live"
        if cap is not None:
            cap.release()


# ── Public: async MJPEG generator ───────────────────────────────────────────
async def live_mjpeg(src: str):
    boundary  = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
    detector  = get_detector()
    auth_map  = _load_auth_map()
    results: Dict[int, dict] = {}   # track_id → finalised plate read
    meta:    Dict[int, dict] = {}   # track_id → {cx, cy, speed, best_crop, …}
    q         = queue.Queue(maxsize=4)
    stop_evt  = threading.Event()
    worker    = threading.Thread(
        target=_ocr_worker, args=(q, results, auth_map, stop_evt), daemon=True)
    worker.start()

    grab = _Grabber(src)
    grab.start()

    frame_idx  = 0
    last_seq   = -1
    fps        = 0.0
    t_prev     = time.time()
    last_status_t = 0.0
    stride     = 1          # adaptive detection stride (1 = detect every frame)
    prev_dets  = []         # last real detection, reused on skip-detect frames
    try:
        while True:
            seq, frame, status = grab.latest()

            # No fresh frame yet → show a status card (throttled) and yield.
            if frame is None or seq == last_seq:
                now = time.time()
                if status != "live" and now - last_status_t > 0.5:
                    last_status_t = now
                    msg = ("Connecting to camera…" if status == "connecting"
                           else "CAMERA OFFLINE")
                    sub = src if status == "connecting" else "Check source / same Wi-Fi"
                    yield boundary + _status_jpg(msg, sub) + b"\r\n"
                await asyncio.sleep(0.01)     # let newer frames arrive / stay responsive
                continue

            last_seq = seq
            frame_idx += 1
            now = time.time()
            dt = now - t_prev
            if dt > 0:
                fps = 0.85 * fps + 0.15 * (1.0 / dt) if fps else (1.0 / dt)
            t_prev = now

            # Adapt stride from measured FPS (hysteresis avoids flip-flopping).
            if fps and fps < _FPS_SLOW:
                stride = _DET_STRIDE_MAX
            elif fps >= _FPS_OK:
                stride = 1
            run_detect = (frame_idx % stride == 0)

            jpg, prev_dets = await asyncio.to_thread(
                _process_frame, detector, frame, frame_idx, results, meta, q, fps,
                run_detect, prev_dets)
            if jpg:
                yield boundary + jpg + b"\r\n"

            # Forget tracks that have left the scene so meta/results don't grow
            # unbounded over a long live session.
            if frame_idx % 150 == 0 and meta:
                for tid in [t for t, m in meta.items()
                            if frame_idx - m["last_seen"] > _PRUNE_AFTER]:
                    meta.pop(tid, None)
                    results.pop(tid, None)
    finally:
        stop_evt.set()
        grab.stop()
