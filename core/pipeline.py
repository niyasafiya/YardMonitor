"""
End-to-end pipeline:

  frame -> detect vehicles (+ track IDs)
        -> on direction-line crossing OR fresh track:
             - crop vehicle -> find plate -> OCR
             - decide() against whitelist
             - actuate gate
             - persist event + push to dashboard

Each camera runs in its own background thread. The pipeline is the bridge
between OpenCV/YOLO and the FastAPI/WebSocket layer.
"""
from __future__ import annotations

import asyncio
import concurrent.futures as _cf
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import numpy as np

from . import config, database as db
from .asset_tracker import AssetTracker
from .detector import Detection, Detector
from .gate_controller import get_gate
from .ocr import PlateOCR

log = logging.getLogger(__name__)

EventCallback = Callable[[dict], None]


@dataclass
class _TrackState:
    """Per-vehicle state to debounce + know which direction it's headed."""
    last_y: int = 0
    crossed: bool = False
    best_plate: Optional[str] = None
    best_conf: float = 0.0
    first_seen: float = field(default_factory=time.time)

    # Authorization state, re-evaluated whenever OCR confidence improves significantly.
    # None = not decided yet, True = whitelisted, False = denied.
    authorized: Optional[bool] = None
    owner: Optional[dict] = None
    decided_at: float = 0.0
    decided_conf: float = 0.0       # confidence at which the last decision was made
    gate_triggered: bool = False    # so we only auto-open the gate once

    # Async OCR — submitted to background thread so frame loop never blocks
    last_ocr_time: float = 0.0
    ocr_future:    Optional[Any] = field(default=None, repr=False)

    # Best plate crop seen across all frames for this track (sharpest wins)
    best_crop:       Optional[Any] = field(default=None, repr=False)
    best_crop_sharp: float = 0.0

    # Set once a crossing event is written to the DB — used to patch it later
    # if OCR improves or authorization changes after the line crossing.
    event_id:         Optional[int]  = None
    event_authorized: Optional[bool] = None


class CameraPipeline(threading.Thread):
    """One camera = one thread = one pipeline."""

    _OCR_INTERVAL   = 0.5   # min seconds between OCR submits per vehicle
    _YOLO_MAX_WIDTH = 320   # resize frame before YOLO; 320 matches imgsz so no internal upscale
    _SHARP_THRESH   = 12.0  # Laplacian variance — lowered to allow slightly blurry crops through

    # Shared single-worker thread pool for OCR across all pipelines.
    # One worker = no GPU/memory contention; OCR is serial by nature.
    _ocr_executor: Optional[_cf.ThreadPoolExecutor] = None
    _executor_lock = threading.Lock()

    @classmethod
    def _get_ocr_executor(cls) -> _cf.ThreadPoolExecutor:
        with cls._executor_lock:
            if cls._ocr_executor is None:
                cls._ocr_executor = _cf.ThreadPoolExecutor(max_workers=1,
                                                            thread_name_prefix="ocr")
            return cls._ocr_executor

    def __init__(self,
                 source_cfg: dict,
                 detector: Detector,
                 ocr: PlateOCR,
                 on_event: Optional[EventCallback] = None,
                 on_frame: Optional[Callable[[str, np.ndarray], None]] = None,
                 on_update: Optional[EventCallback] = None):
        super().__init__(daemon=True, name=f"cam-{source_cfg['id']}")
        self.cfg = source_cfg
        self.id: str = source_cfg["id"]
        self.name: str = source_cfg.get("name", self.id)
        self.role: str = source_cfg.get("role", "gate")
        self.uri = source_cfg["uri"]
        self.fps_target = int(source_cfg.get("fps_target", 8))
        self.line_y_frac = float(source_cfg.get("direction_line_y", 0.55))

        self.detector = detector
        self.ocr = ocr
        self.on_event = on_event
        self.on_frame = on_frame
        self.on_update = on_update

        self.stopping = threading.Event()
        self._tracks: dict[int, _TrackState] = {}
        self._dup_window = int(config.get("thresholds", "duplicate_window_sec", default=30))
        self.snapshot_dir = Path(config.get("storage", "snapshot_dir", default="data/snapshots"))
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

        self.asset_tracker: Optional[AssetTracker] = None
        if self.role == "yard":
            self.asset_tracker = AssetTracker(
                camera_id=self.id,
                snapshot_dir=str(self.snapshot_dir / "assets"),
            )

        self.last_frame_jpeg: Optional[bytes] = None
        self._frame_lock = threading.Lock()
        self._pending_uri = None   # set by switch_source() to trigger a live swap

    # ------------------------------------------------------------------

    def stop(self):
        self.stopping.set()

    def switch_source(self, new_uri) -> None:
        """Request a live camera source swap (thread-safe). Takes effect on next loop tick."""
        self._pending_uri = new_uri
        log.info("Camera %s: source switch queued → %s", self.id, new_uri)

    def get_jpeg(self) -> Optional[bytes]:
        with self._frame_lock:
            return self.last_frame_jpeg

    # ------------------------------------------------------------------

    def run(self):
        log.info("Camera %s starting on %s", self.id, self.uri)

        # Keep retrying until the camera is available (e.g. iVCam still launching)
        cap = None
        retry_delay = 3
        while not self.stopping.is_set():
            cap = self._open_capture()
            if cap is not None:
                log.info("Camera %s: source opened successfully", self.id)
                break
            log.warning(
                "Camera %s: could not open source %s — retrying in %ds "
                "(make sure iVCam PC client is running)",
                self.id, self.uri, retry_delay,
            )
            time.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, 30)  # back off up to 30s

        if cap is None:
            return

        interval = 1.0 / max(self.fps_target, 1)
        last_proc = 0.0

        try:
            while not self.stopping.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    # Loop video files; reconnect streams
                    if isinstance(self.uri, str) and self.uri.endswith((".mp4", ".avi", ".mov", ".mkv")):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    log.warning("Camera %s: read failed — reconnecting in 3s", self.id)
                    time.sleep(3)
                    cap.release()
                    cap = self._open_capture()
                    if cap is None:
                        log.warning("Camera %s: reconnect failed — will retry in 5s", self.id)
                        time.sleep(5)
                    continue

                # Live source switch requested from the API
                if self._pending_uri is not None:
                    new_uri = self._pending_uri
                    self._pending_uri = None
                    log.info("Camera %s: switching source %s → %s", self.id, self.uri, new_uri)
                    cap.release()
                    self.uri = new_uri
                    cap = self._open_capture()
                    if cap is None:
                        log.warning("Camera %s: new source %s not available — retrying", self.id, new_uri)
                    continue

                now = time.time()
                if now - last_proc < interval:
                    time.sleep(0.005)   # yield CPU instead of busy-spinning
                    continue
                last_proc = now

                annotated = self._process_frame(frame)
                self._publish_frame(annotated)
        finally:
            try:
                cap.release()
            except Exception:
                pass
            log.info("Camera %s stopped", self.id)

    # ------------------------------------------------------------------

    def _open_capture(self):
        uri = self.uri
        # Allow integer webcam IDs in YAML
        try:
            uri_int = int(uri)
            cap = cv2.VideoCapture(uri_int)
        except (TypeError, ValueError):
            cap = cv2.VideoCapture(uri)

        if not cap.isOpened():
            return None
        # Lower internal buffer for live streams (cuts latency a lot)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        return cap

    # ------------------------------------------------------------------

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        line_y = int(h * self.line_y_frac)

        # Downscale for YOLO inference — halves pixel count on 1280px streams
        if w > self._YOLO_MAX_WIDTH:
            scale_down = self._YOLO_MAX_WIDTH / w
            small      = cv2.resize(frame,
                                    (self._YOLO_MAX_WIDTH, int(h * scale_down)),
                                    interpolation=cv2.INTER_AREA)
            scale_up = 1.0 / scale_down
        else:
            small    = frame
            scale_up = 1.0

        detections = self.detector.detect_and_track(small)

        # Scale bounding boxes back to original resolution for annotation/OCR
        if scale_up != 1.0:
            for d in detections:
                x1, y1, x2, y2 = d.bbox
                d.bbox = (int(x1 * scale_up), int(y1 * scale_up),
                          int(x2 * scale_up), int(y2 * scale_up))

        annotated = frame.copy()
        # Draw the direction line on gate cameras
        if self.role == "gate":
            cv2.line(annotated, (0, line_y), (w, line_y), (0, 200, 255), 2)
            cv2.putText(annotated, "DIRECTION LINE", (10, line_y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

        for det in detections:
            if self.role == "gate":
                self._handle_gate_detection(frame, det, line_y)
            self._draw_detection(annotated, det)

        if self.asset_tracker is not None:
            self.asset_tracker.update(frame, detections)

        # Top-left camera label
        cv2.rectangle(annotated, (0, 0), (300, 30), (0, 0, 0), -1)
        cv2.putText(annotated,
                    f"{self.name}  |  {len(detections)} obj",
                    (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        # Top-right gate HUD
        if self.role == "gate":
            self._draw_gate_hud(annotated)
            self._draw_access_banner(annotated)

        return annotated

    # ------------------------------------------------------------------

    def _handle_gate_detection(self, frame: np.ndarray, det: Detection, line_y: int):
        if det.track_id is None:
            return
        tid = det.track_id
        st  = self._tracks.get(tid)
        if st is None:
            st = _TrackState(last_y=det.center[1])
            self._tracks[tid] = st

        # ---- Collect finished async OCR result (never blocks) ----
        if st.ocr_future is not None and st.ocr_future.done():
            try:
                plate, conf = st.ocr_future.result()
                if plate and conf > st.best_conf:
                    st.best_plate = plate
                    st.best_conf  = conf
                    log.debug("OCR result for track %s: %s (%.2f)", tid, plate, conf)
                    # If the crossing event was already logged, update it with the
                    # improved plate read so the dashboard log stays accurate.
                    if st.event_id is not None:
                        self._patch_logged_event(st)
            except Exception as exc:
                log.debug("OCR future error: %s", exc)
            finally:
                st.ocr_future = None

        # ---- Always track the sharpest plate crop seen so far ----
        crop = self._get_plate_crop(frame, det)
        if crop is not None:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
            sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            if sharpness > st.best_crop_sharp:
                st.best_crop_sharp = sharpness
                st.best_crop = crop.copy()

        # ---- Submit a new OCR job if no job is running and we need one ----
        if (st.best_plate is None or st.best_conf < 0.80) and st.ocr_future is None:
            now = time.time()
            if now - st.last_ocr_time >= self._OCR_INTERVAL:
                if st.best_crop is not None and st.best_crop_sharp >= self._SHARP_THRESH:
                    st.last_ocr_time = now
                    st.ocr_future = self._get_ocr_executor().submit(
                        self.ocr.read, st.best_crop.copy()   # best crop so far, not just this frame
                    )

        # ---- Decide access when we have a plate read, or re-decide if OCR
        # improved significantly (catches low-conf misreads like K→0, L→4).
        # DENY is re-evaluated whenever confidence jumps ≥20 pts above the
        # confidence that produced the previous (wrong) decision.
        _should_decide = (
            st.best_plate is not None
            and st.best_conf >= 0.35
            and (
                st.authorized is None
                or (st.authorized is False
                    and st.best_conf >= st.decided_conf + 0.20)
            )
        )
        if _should_decide:
            owner = db.lookup_plate(st.best_plate)

            # Fuzzy fallback: OCR sometimes swaps visually-similar chars
            # (K↔C, L↔K, 4↔2). If no exact match, check whether the read
            # plate is ≥65% positional-character match to a whitelist plate.
            # Only fires when confidence is reasonable and the match is unambiguous.
            if owner is None and st.best_conf >= 0.30:
                min_sim = float(config.get("thresholds", "plate_fuzzy_similarity",
                                           default=0.60))
                fuzzy_owner = db.fuzzy_lookup_plate(st.best_plate, min_similarity=min_sim)
                if fuzzy_owner:
                    log.info("OCR corrected by fuzzy match: '%s' → '%s'",
                             st.best_plate, fuzzy_owner["plate"])
                    st.best_plate = fuzzy_owner["plate"]   # use the authoritative plate
                    owner = fuzzy_owner

            if owner:
                st.authorized = True
                st.owner = owner
                st.decided_at = time.time()
                st.decided_conf = st.best_conf
                # Auto-open the gate the SECOND a known plate is recognized,
                # without waiting for the line crossing. We still record the
                # crossing event separately below for entry/exit accounting.
                if not st.gate_triggered:
                    st.gate_triggered = True
                    gate = get_gate()
                    decision = gate.decide(st.best_plate)
                    if decision.will_open:
                        gate.actuate(decision)
                        log.info("GATE auto-open for %s (owner=%s)",
                                 st.best_plate, owner.get("owner_name"))
            else:
                st.authorized = False
                st.decided_at = time.time()
                st.decided_conf = st.best_conf

            # If the crossing event was already logged with a different outcome,
            # patch it now so the dashboard reflects the correct decision.
            if st.event_id is not None and st.authorized != st.event_authorized:
                self._patch_logged_event(st)

        cy = det.center[1]
        crossed_down = st.last_y < line_y <= cy        # entry
        crossed_up = st.last_y > line_y >= cy          # exit
        st.last_y = cy

        if (crossed_down or crossed_up) and not st.crossed:
            st.crossed = True
            direction = "entry" if crossed_down else "exit"
            self._emit_event(frame, det, st, direction)

    # ------------------------------------------------------------------

    def _get_plate_crop(self, frame: np.ndarray, det: Detection) -> Optional[np.ndarray]:
        """Extract the plate region from a vehicle detection. Returns None if unusable."""
        x1, y1, x2, y2 = det.bbox
        x1 = max(0, x1);  y1 = max(0, y1)
        x2 = min(frame.shape[1], x2); y2 = min(frame.shape[0], y2)
        if x2 - x1 < 30 or y2 - y1 < 30:
            return None
        veh_crop  = frame[y1:y2, x1:x2]
        plate_box = self.detector.find_plate(veh_crop)
        if plate_box is not None:
            px1, py1, px2, py2 = plate_box
            # Add margin so edge characters aren't clipped by a tight detector box
            mx = max(6, int((px2 - px1) * 0.06))
            my = max(4, int((py2 - py1) * 0.10))
            px1 = max(0, px1 - mx);  py1 = max(0, py1 - my)
            px2 = min(veh_crop.shape[1], px2 + mx)
            py2 = min(veh_crop.shape[0], py2 + my)
            box_w = px2 - px1
            box_h = py2 - py1
            # A real plate is always wider than tall (≥2:1 ratio). A nearly-square
            # or tall box means the detector found a sub-region — discard it and
            # fall through to the bottom-strip fallback below.
            if box_w >= 20 and box_h >= 5 and box_w >= 2.0 * box_h:
                return CameraPipeline._dewarp_plate(veh_crop[py1:py2, px1:px2])

        # Plate detector missed or returned a bad box — fall back to the bottom
        # 30% of the vehicle crop (where license plates always sit).
        # Use center 85% width to avoid vehicle side edges.
        vh, vw = veh_crop.shape[:2]
        fx1 = int(vw * 0.075)
        fx2 = int(vw * 0.925)
        fallback = veh_crop[int(vh * 0.70):, fx1:fx2]
        if fallback.shape[0] >= 8 and fallback.shape[1] >= 30:
            return CameraPipeline._dewarp_plate(fallback)
        return None

    @staticmethod
    def _is_sharp_enough(crop: np.ndarray) -> bool:
        """Reject blurry/empty crops before sending to OCR (fast Laplacian check)."""
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        return float(cv2.Laplacian(gray, cv2.CV_64F).var()) >= CameraPipeline._SHARP_THRESH

    @staticmethod
    def _dewarp_plate(crop: np.ndarray) -> np.ndarray:
        """Perspective-dewarp a plate crop to a flat rectangle.

        Finds the plate quadrilateral via edge detection + contour analysis and
        applies a 4-point perspective transform.  Falls back to the original crop
        if a clean 4-corner polygon cannot be found — never makes things worse.
        """
        h, w = crop.shape[:2]
        if w < 40 or h < 8:
            return crop

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop.copy()

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        gray = clahe.apply(gray)

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 100)

        # Horizontal dilation merges character blobs into one plate-shaped blob
        kw = max(10, w // 10)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 3))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

        best_quad = None
        best_area = 0.0
        min_area = w * h * 0.10  # must cover ≥10% of crop to be valid

        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area:
                continue
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.05 * peri, True)
            if len(approx) != 4:
                continue
            # Reject tall/square shapes — a real plate is always wider than tall
            rx, ry, rw, rh = cv2.boundingRect(approx)
            if rw < 1.5 * rh:
                continue
            if area > best_area:
                best_area = area
                best_quad = approx

        if best_quad is None:
            return crop

        # Order corners: top-left, top-right, bottom-right, bottom-left
        # Classic "sum / diff" trick: TL=min(x+y), BR=max(x+y), TR=min(y-x), BL=max(y-x)
        pts = best_quad.reshape(4, 2).astype(np.float32)
        s = pts.sum(axis=1)
        d = pts[:, 1] - pts[:, 0]
        src = np.array([pts[np.argmin(s)],   # TL
                        pts[np.argmin(d)],   # TR
                        pts[np.argmax(s)],   # BR
                        pts[np.argmax(d)]], dtype=np.float32)  # BL

        out_w = max(400, w)
        out_h = max(100, h)
        dst = np.array([[0, 0], [out_w, 0], [out_w, out_h], [0, out_h]],
                       dtype=np.float32)

        M = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(crop, M, (out_w, out_h),
                                   flags=cv2.INTER_LANCZOS4)

    # ------------------------------------------------------------------

    def _patch_logged_event(self, st: _TrackState) -> None:
        """Update a crossing event already in the DB with improved OCR / auth data."""
        if st.event_id is None:
            return
        plate = st.best_plate
        conf  = st.best_conf

        owner = db.lookup_plate(plate) if plate else None
        if owner is None and plate and conf >= 0.30:
            min_sim = float(config.get("thresholds", "plate_fuzzy_similarity", default=0.60))
            owner   = db.fuzzy_lookup_plate(plate, min_similarity=min_sim)
        authorized = owner is not None

        updated = db.update_event(
            st.event_id,
            plate=plate,
            plate_confidence=conf,
            authorized=authorized,
            gate_opened=st.gate_triggered and authorized,
        )
        if updated is None:
            return

        st.event_authorized = authorized
        log.info("Event %d updated: plate=%s auth=%s", st.event_id, plate, authorized)

        if self.on_update:
            payload = dict(updated)
            payload["owner"]       = owner
            payload["camera_name"] = self.name
            try:
                self.on_update(payload)
            except Exception as e:
                log.warning("on_update callback failed: %s", e)

    # ------------------------------------------------------------------

    def _emit_event(self, frame: np.ndarray, det: Detection,
                    st: _TrackState, direction: str):
        plate = st.best_plate
        conf = st.best_conf

        # Debounce: ignore the same plate seen N seconds ago
        if plate:
            recent = db.recent_event_for_plate(plate, self._dup_window)
            if recent:
                log.debug("Debounced duplicate event for plate %s", plate)
                return

        # Snapshot — write to disk but store ONLY the basename so the URL
        # built by the frontend (`/snapshots/<basename>`) is OS-independent.
        snap_name = f"{int(time.time())}_{self.id}_{plate or 'unknown'}.jpg"
        disk_path = str(self.snapshot_dir / snap_name)
        snap_path: Optional[str] = snap_name
        try:
            cv2.imwrite(disk_path, frame)
        except Exception:
            snap_path = None

        # Gate decision
        gate = get_gate()
        decision = gate.decide(plate)
        if decision.will_open:
            gate.actuate(decision)

        ev = db.add_event(
            timestamp=datetime.utcnow(),
            camera_id=self.id,
            plate=plate,
            plate_confidence=conf,
            direction=direction,
            authorized=decision.authorized,
            gate_opened=decision.will_open,
            snapshot_path=snap_path,
            vehicle_type=det.cls_name,
            track_id=det.track_id,
            notes=decision.reason,
        )

        # Remember this event's DB id so we can patch it if OCR improves later.
        st.event_id         = ev["id"]
        st.event_authorized = decision.authorized

        log.info(
            "EVENT cam=%s plate=%s dir=%s auth=%s opened=%s reason=%s",
            self.id, plate, direction, decision.authorized,
            decision.will_open, decision.reason,
        )

        if self.on_event:
            payload = dict(ev)
            payload["owner"]       = decision.owner
            payload["camera_name"] = self.name
            payload["gate_is_open"]= decision.will_open   # real-time gate state hint
            try:
                self.on_event(payload)
            except Exception as e:
                log.warning("on_event callback failed: %s", e)

    # ------------------------------------------------------------------

    # ---- BGR color palette (OpenCV uses BGR not RGB) ---------------------
    _C_AUTH    = (94, 220, 86)    # green
    _C_DENIED  = (87, 95, 255)    # red
    _C_READING = (60, 200, 245)   # amber
    _C_TRACK   = (255, 180, 90)   # blue (no plate yet)

    def _track_color(self, st: Optional["_TrackState"]) -> tuple[int, int, int]:
        if st is None:
            return self._C_TRACK
        if st.authorized is True:  return self._C_AUTH
        if st.authorized is False: return self._C_DENIED
        if st.best_plate:          return self._C_READING
        return self._C_TRACK

    def _draw_detection(self, frame: np.ndarray, det: Detection):
        x1, y1, x2, y2 = det.bbox
        st = self._tracks.get(det.track_id) if det.track_id is not None else None
        color = self._track_color(st)

        # Vehicle bounding box (thicker when decided)
        thickness = 3 if (st and st.authorized is not None) else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # Status text above the box: AUTHORIZED / DENIED / SCANNING / TRACKING
        if st and st.authorized is True:
            status_txt = "AUTHORIZED"
        elif st and st.authorized is False:
            status_txt = "DENIED"
        elif st and st.best_plate:
            status_txt = "SCANNING…"
        else:
            status_txt = "TRACKING"

        plate_txt = st.best_plate if st and st.best_plate else ""
        line1 = f"{status_txt}" + (f"  {plate_txt}" if plate_txt else "")
        line2 = f"{det.cls_name} {det.confidence:.2f}" + (
            f"  id={det.track_id}" if det.track_id is not None else ""
        )

        # Background pill for status line (top)
        (w1, h1), _ = cv2.getTextSize(line1, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - h1 - 12), (x1 + w1 + 14, y1), color, -1)
        cv2.putText(frame, line1, (x1 + 7, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (10, 10, 10), 2)

        # Sub-label (smaller, bottom of box)
        (w2, h2), _ = cv2.getTextSize(line2, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(frame, (x1, y2), (x1 + w2 + 10, y2 + h2 + 8), color, -1)
        cv2.putText(frame, line2, (x1 + 5, y2 + h2 + 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (10, 10, 10), 1)

    # ------------------------------------------------------------------
    # HUD + banners (gate cameras only)
    # ------------------------------------------------------------------

    def _draw_gate_hud(self, frame: np.ndarray):
        """Small top-right pill showing current gate state."""
        h, w = frame.shape[:2]
        is_open = get_gate().is_open
        label = "GATE: OPEN" if is_open else "GATE: CLOSED"
        color = self._C_AUTH if is_open else (90, 90, 90)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        pad = 10
        x2 = w - 10; x1 = x2 - tw - pad * 2
        y1 = 8; y2 = y1 + th + pad
        # translucent backdrop
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1 + pad, y2 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    def _draw_access_banner(self, frame: np.ndarray):
        """Big translucent banner bottom-center, lasts ~3s per decision."""
        # Pick the freshest decided track
        now = time.time()
        recent = [(tid, st) for tid, st in self._tracks.items()
                  if st.authorized is not None and now - st.decided_at < 3.0]
        if not recent:
            return
        recent.sort(key=lambda kv: kv[1].decided_at, reverse=True)
        _, st = recent[0]

        h, w = frame.shape[:2]
        if st.authorized:
            color = self._C_AUTH
            title = "ACCESS GRANTED"
            owner_name = (st.owner or {}).get("owner_name") if st.owner else None
            subline = f"{st.best_plate}" + (f"  •  {owner_name}" if owner_name else "")
        else:
            color = self._C_DENIED
            title = "ACCESS DENIED"
            subline = f"{st.best_plate}  •  not whitelisted"

        # Banner geometry — bottom 18% of frame, full width
        bh = max(80, int(h * 0.18))
        y1 = h - bh - 12
        y2 = h - 12
        x1 = 12
        x2 = w - 12

        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

        # Title (huge)
        (tw, th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_DUPLEX, 1.6, 3)
        cv2.putText(frame, title,
                    (x1 + (x2 - x1 - tw) // 2, y1 + th + 14),
                    cv2.FONT_HERSHEY_DUPLEX, 1.6, color, 3)

        # Subline (plate + owner)
        (sw, sh), _ = cv2.getTextSize(subline, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.putText(frame, subline,
                    (x1 + (x2 - x1 - sw) // 2, y2 - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (235, 235, 235), 2)

    def _publish_frame(self, frame: np.ndarray):
        # JPEG-encode for MJPEG stream / WebSocket previews
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 55])
        if ok:
            with self._frame_lock:
                self.last_frame_jpeg = buf.tobytes()
        if self.on_frame:
            try:
                self.on_frame(self.id, frame)
            except Exception:
                pass


# ----------------------------------------------------------------------------
# Pipeline orchestrator
# ----------------------------------------------------------------------------

class PipelineManager:
    """Owns one CameraPipeline per configured source."""

    def __init__(self, on_event: Optional[EventCallback] = None,
                 on_update: Optional[EventCallback] = None):
        self.on_event  = on_event
        self.on_update = on_update
        self.detector: Optional[Detector] = None
        self.ocr: Optional[PlateOCR] = None
        self.pipelines: dict[str, CameraPipeline] = {}

    def start(self):
        log.info("Loading models...")
        self.detector = Detector()
        gpu = bool(int(os.getenv("YM_USE_GPU", "0")))
        self.ocr = PlateOCR(gpu=gpu)

        for src in config.get("sources", default=[]):
            cp = CameraPipeline(src, self.detector, self.ocr,
                                on_event=self.on_event, on_update=self.on_update)
            cp.start()
            self.pipelines[cp.id] = cp
            log.info("Started pipeline for camera %s", cp.id)

    def stop(self):
        for cp in self.pipelines.values():
            cp.stop()
        for cp in self.pipelines.values():
            cp.join(timeout=2)

    def get_jpeg(self, camera_id: str) -> Optional[bytes]:
        cp = self.pipelines.get(camera_id)
        return cp.get_jpeg() if cp else None

    def cameras(self) -> list[dict]:
        return [
            {"id": cp.id, "name": cp.name, "role": cp.role, "uri": cp.uri}
            for cp in self.pipelines.values()
        ]

    def switch_camera_source(self, camera_id: str, new_uri) -> bool:
        cp = self.pipelines.get(camera_id)
        if cp is None:
            return False
        cp.switch_source(new_uri)
        return True

    def start_demo(self, video_path: str) -> str:
        """Start (or restart) a demo camera pipeline from a video file."""
        self.stop_demo()   # stop any existing demo first
        src = {
            "id":               "demo_cam",
            "name":             "Demo Video",
            "role":             "gate",
            "uri":              video_path,
            "fps_target":       8,
            "direction_line_y": 0.55,
        }
        cp = CameraPipeline(src, self.detector, self.ocr,
                            on_event=self.on_event, on_update=self.on_update)
        cp.start()
        self.pipelines["demo_cam"] = cp
        log.info("Demo pipeline started: %s", video_path)
        return "demo_cam"

    def stop_demo(self) -> bool:
        """Stop and remove the demo camera pipeline if it is running."""
        cp = self.pipelines.pop("demo_cam", None)
        if cp:
            cp.stop()
            cp.join(timeout=3)
            log.info("Demo pipeline stopped")
            return True
        return False

