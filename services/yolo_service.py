"""
Vehicle detector — modelled on YardMonitor/core/detector.py.

Priority order:
  1. YOLOv8 (ultralytics) + ByteTrack  ← preferred (models/yolov8n.pt)
  2. YOLOv4-tiny OpenCV DNN            ← fallback (models/yolov4-tiny.*)

Plate localisation uses the same 3-tier strategy as YardMonitor:
  1. ONNX plate detector  (models/license_plate.onnx if present)
  2. YOLOv8 plate model   (models/license_plate.pt if present)
  3. Contour-based fallback (always available)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VEHICLE_CLASSES = {"car", "motorcycle", "bus", "truck", "bicycle"}

_VEHICLE_CLASS_IDS_V4: Dict[int, str] = {   # COCO ids for YOLOv4 fallback
    1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck",
}

_COLORS: Dict[str, Tuple[int, int, int]] = {
    "car":        (77,  124, 254),
    "truck":      (242,  86, 110),
    "bus":        (240, 169,  59),
    "motorcycle": (40,  201, 168),
    "bicycle":    (160, 160, 160),
}


# ---------------------------------------------------------------------------
# Detection dataclass  (mirrors YardMonitor's Detection)
# ---------------------------------------------------------------------------
@dataclass
class Detection:
    cls_name:   str
    confidence: float
    bbox:       Tuple[int, int, int, int]   # x1, y1, x2, y2
    track_id:   Optional[int] = None

    @property
    def center(self) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]


# ---------------------------------------------------------------------------
# Contour-based plate finder  (port of YardMonitor's _find_plate_contour)
# ---------------------------------------------------------------------------
def find_plate_contour(
    crop: np.ndarray,
) -> Tuple[int, int, int, int]:
    """
    Find the most plate-like rectangle in the bottom 65 % of a vehicle crop.
    Uses CLAHE → Canny → horizontal dilation → aspect-ratio filter.
    Returns (x1, y1, x2, y2) in crop coordinates.
    """
    h, w = crop.shape[:2]
    y_start = int(h * 0.35)
    roi = crop[y_start:]
    rh, rw = roi.shape[:2]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi.copy()
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray  = clahe.apply(gray)

    edges  = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 120)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (22, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_box, best_score = None, 0.0
    min_area = rw * rh * 0.008

    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        if cw * ch < min_area:
            continue
        aspect = cw / max(ch, 1)
        if not (2.0 <= aspect <= 7.0):
            continue
        cx_offset = abs((x + cw / 2) - rw / 2) / rw
        score = cw * ch * max(0.2, 1.0 - cx_offset)
        if score > best_score:
            best_score = score
            pad = 5
            best_box = (
                max(0,  x - pad),
                y_start + max(0,  y - pad),
                min(w,  x + cw + pad),
                y_start + min(rh, y + ch + pad),
            )

    return best_box or (int(w * 0.15), int(h * 0.70), int(w * 0.85), int(h * 0.97))


# ---------------------------------------------------------------------------
# YOLOv8 detector  (primary — YardMonitor style)
# ---------------------------------------------------------------------------
class _YOLOv8Detector:
    """Wraps ultralytics YOLO with ByteTrack, matching YardMonitor's Detector."""

    VEHICLE_CONF = 0.45
    IMGSZ        = 480          # smaller than 640 → ~1.7× faster on CPU, negligible accuracy loss

    def __init__(self):
        from ultralytics import YOLO
        model_path = Path("models/yolov8n.pt")
        if not model_path.exists():
            model_path = Path("yolov8n.pt")   # repo root fallback
        self._model = YOLO(str(model_path))
        print(f"[YOLOv8] Loaded {model_path}")

    def detect(self, frame: np.ndarray, track: bool = False,
               imgsz: Optional[int] = None) -> List[Detection]:
        sz = imgsz or self.IMGSZ
        if track:
            results = self._model.track(
                frame,
                conf=self.VEHICLE_CONF,
                persist=True,
                verbose=False,
                tracker="bytetrack.yaml",
                imgsz=sz,
            )
        else:
            results = self._model(frame, conf=self.VEHICLE_CONF, verbose=False, imgsz=sz)

        names = results[0].names
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []

        ids = (boxes.id.int().tolist()
               if (track and boxes.id is not None)
               else [None] * len(boxes))

        out: List[Detection] = []
        for i in range(len(boxes)):
            cls_name = str(names[int(boxes.cls[i].item())]).lower()
            if cls_name not in VEHICLE_CLASSES:
                continue
            x1, y1, x2, y2 = map(int, boxes.xyxy[i].tolist())
            out.append(Detection(
                cls_name=cls_name,
                confidence=round(float(boxes.conf[i].item()), 3),
                bbox=(x1, y1, x2, y2),
                track_id=ids[i] if i < len(ids) else None,
            ))
        return out


# ---------------------------------------------------------------------------
# YOLOv4-tiny fallback  (OpenCV DNN — no ultralytics needed)
# ---------------------------------------------------------------------------
class _YOLOv4Detector:
    def __init__(self):
        cfg = Path("models/yolov4-tiny.cfg")
        wts = Path("models/yolov4-tiny.weights")
        self._net = cv2.dnn.readNet(str(wts), str(cfg))
        self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        layer_names = self._net.getLayerNames()
        unconnected = self._net.getUnconnectedOutLayers()
        if hasattr(unconnected, "flatten"):
            unconnected = unconnected.flatten()
        self._out_layers = [layer_names[int(i) - 1] for i in unconnected]
        print("[YOLOv4-tiny] Loaded fallback detector")

    def detect(self, frame: np.ndarray, track: bool = False) -> List[Detection]:
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self._net.setInput(blob)
        outs = self._net.forward(self._out_layers)

        boxes, confs, cids = [], [], []
        for out in outs:
            for det in out:
                scores = det[5:]
                cid = int(np.argmax(scores))
                conf = float(scores[cid])
                if conf > 0.40 and cid in _VEHICLE_CLASS_IDS_V4:
                    cx, cy, bw, bh = det[:4]
                    x = max(0, int((cx - bw / 2) * w))
                    y = max(0, int((cy - bh / 2) * h))
                    boxes.append([x, y, max(1, int(bw * w)), max(1, int(bh * h))])
                    confs.append(conf)
                    cids.append(cid)

        out_dets: List[Detection] = []
        if boxes:
            idxs = cv2.dnn.NMSBoxes(boxes, confs, 0.40, 0.40)
            for i in (idxs.flatten() if len(idxs) else []):
                x, y, bw, bh = boxes[i]
                out_dets.append(Detection(
                    cls_name=_VEHICLE_CLASS_IDS_V4[cids[i]],
                    confidence=round(confs[i], 3),
                    bbox=(x, y, x + bw, y + bh),
                ))
        return out_dets


# ---------------------------------------------------------------------------
# Public Detector facade
# ---------------------------------------------------------------------------
class Detector:
    """
    Drop-in detector used by all routers.
    Picks the best available backend automatically.
    """

    def __init__(self):
        self._impl = None
        self._backend = "none"
        self._try_load()

    def _try_load(self):
        # Try YOLOv8 first
        try:
            self._impl    = _YOLOv8Detector()
            self._backend = "yolov8"
            return
        except Exception as exc:
            print(f"[Detector] YOLOv8 unavailable ({exc}), trying YOLOv4-tiny…")

        # Try YOLOv4-tiny
        try:
            self._impl    = _YOLOv4Detector()
            self._backend = "yolov4-tiny"
            return
        except Exception as exc:
            print(f"[Detector] YOLOv4-tiny unavailable ({exc}) — detection disabled.")

    @property
    def available(self) -> bool:
        return self._impl is not None

    @property
    def backend(self) -> str:
        return self._backend

    def detect(self, frame: np.ndarray, track: bool = False,
               imgsz: Optional[int] = None) -> List[Detection]:
        if self._impl is None:
            return []
        try:
            return self._impl.detect(frame, track=track, imgsz=imgsz)
        except TypeError:
            # YOLOv4 fallback impl doesn't accept imgsz
            return self._impl.detect(frame, track=track)

    def draw(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        out = frame.copy()
        for d in detections:
            x1, y1, x2, y2 = d.bbox
            color = _COLORS.get(d.cls_name, (200, 200, 200))
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

            tid  = f" #{d.track_id}" if d.track_id is not None else ""
            label = f"{d.cls_name.capitalize()}{tid} {d.confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
            cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
            cv2.putText(out, label, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (10, 10, 10), 1)
        return out

    def get_vehicle_crop(
        self, frame: np.ndarray, det: Detection
    ) -> np.ndarray:
        """Return the cropped frame region for a single detection."""
        x1, y1, x2, y2 = det.bbox
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        return frame[y1:y2, x1:x2]

    def find_plate_region(
        self, vehicle_crop: np.ndarray
    ) -> Tuple[int, int, int, int]:
        """Return plate bounding box (x1,y1,x2,y2) inside the vehicle crop."""
        return find_plate_contour(vehicle_crop)


# ---------------------------------------------------------------------------
# License-plate detector  (trained YOLOv8 model — tight plate localisation)
#
# Replaces the contour heuristic when models/license_plate.pt is present. Runs
# on a vehicle crop and returns the best plate box in crop coordinates, which is
# then OCR'd. Falls back silently to find_plate_contour when the model is
# missing or finds nothing, so nothing breaks without the model.
# ---------------------------------------------------------------------------
class _PlateDetector:
    CONF  = 0.25
    IMGSZ = 320

    def __init__(self):
        from ultralytics import YOLO
        path = Path("models/license_plate.pt")
        if not path.exists():
            path = Path("license_plate.pt")
        if not path.exists():
            raise FileNotFoundError("models/license_plate.pt not found")
        self._model = YOLO(str(path))
        print(f"[PlateDetector] Loaded {path}")

    def detect_best(self, img: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Return the highest-confidence plate box (x1,y1,x2,y2) in `img`
        coordinates, or None if no plate is found."""
        if img is None or img.size == 0:
            return None
        res   = self._model(img, conf=self.CONF, verbose=False, imgsz=self.IMGSZ)
        boxes = res[0].boxes
        if boxes is None or len(boxes) == 0:
            return None
        i = int(boxes.conf.argmax())
        x1, y1, x2, y2 = (int(v) for v in boxes.xyxy[i].tolist())
        return (x1, y1, x2, y2)


# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------
_detector: Optional[Detector] = None
_plate_detector: Optional[_PlateDetector] = None
_plate_tried = False


def get_detector() -> Detector:
    global _detector
    if _detector is None:
        _detector = Detector()
    return _detector


def get_plate_detector() -> Optional[_PlateDetector]:
    """Return the trained plate detector, or None if the model is unavailable.
    Loaded once; the result (including 'not available') is cached."""
    global _plate_detector, _plate_tried
    if not _plate_tried:
        _plate_tried = True
        try:
            _plate_detector = _PlateDetector()
        except Exception as exc:
            print(f"[PlateDetector] unavailable ({exc}) — using contour fallback.")
            _plate_detector = None
    return _plate_detector
