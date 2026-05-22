"""
Vehicle + license-plate detection using YOLOv8 (ultralytics).

We use two models:

* `vehicle_detector`: standard YOLOv8n (COCO classes). Auto-downloads.
* `plate_detector`:   YOLO model trained on license plates. Download via
                      `python scripts/download_models.py` or supply your own.

If the plate model is missing, we gracefully fall back to running OCR over the
bottom half of the vehicle bounding box, which works on most clear footage.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from . import config

log = logging.getLogger(__name__)


@dataclass
class Detection:
    cls_name: str
    confidence: float
    bbox: tuple[int, int, int, int]   # x1, y1, x2, y2
    track_id: Optional[int] = None    # set by tracker

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]

    @property
    def center(self) -> tuple[int, int]:
        return ((self.bbox[0] + self.bbox[2]) // 2,
                (self.bbox[1] + self.bbox[3]) // 2)


class Detector:
    """Wrapper around ultralytics YOLO with sensible defaults."""

    def __init__(self):
        # Lazy import — ultralytics is heavy.
        from ultralytics import YOLO

        self._YOLO = YOLO
        vehicle_path = config.get("models", "vehicle_detector", default="models/yolov8n.pt")
        plate_path = config.get("models", "plate_detector", default="models/license_plate.pt")

        self.vehicle_model = YOLO(vehicle_path)
        log.info("Loaded vehicle detector: %s", vehicle_path)

        self.plate_model = None
        if Path(plate_path).exists():
            try:
                self.plate_model = YOLO(plate_path)
                log.info("Loaded plate detector: %s", plate_path)
            except Exception as e:
                log.warning("Could not load plate detector %s: %s", plate_path, e)
        else:
            log.warning(
                "Plate detector not found at %s. Falling back to vehicle-crop OCR. "
                "Run `python scripts/download_models.py` to fetch one.",
                plate_path,
            )

        self.vehicle_classes = set(
            c.lower() for c in config.get("models", "vehicle_classes",
                                          default=["car", "motorcycle", "bus", "truck"])
        )
        self.asset_classes = set(
            c.lower() for c in config.get("models", "asset_classes",
                                          default=["car", "truck", "bus", "motorcycle"])
        )
        self.vehicle_conf = float(config.get("thresholds", "vehicle_conf", default=0.45))
        self.plate_conf = float(config.get("thresholds", "plate_conf", default=0.35))

    # ------------------------------------------------------------------
    # Detection + tracking
    # ------------------------------------------------------------------

    def detect_and_track(self, frame: np.ndarray, persist: bool = True) -> List[Detection]:
        """
        Detect vehicles AND assign persistent track IDs across frames.

        Uses ultralytics' built-in ByteTrack tracker (`model.track`).
        """
        results = self.vehicle_model.track(
            frame,
            conf=self.vehicle_conf,
            persist=persist,
            verbose=False,
            tracker="bytetrack.yaml",
            imgsz=320,   # match _YOLO_MAX_WIDTH — avoids internal upscale to 640
        )
        return self._yolo_to_detections(results[0], filter_classes=self.vehicle_classes)

    def detect_assets(self, frame: np.ndarray) -> List[Detection]:
        """Detect anything we treat as an asset (no tracking — yard cam)."""
        results = self.vehicle_model.track(
            frame, conf=self.vehicle_conf, persist=True,
            verbose=False, tracker="bytetrack.yaml",
            imgsz=320,
        )
        return self._yolo_to_detections(results[0], filter_classes=self.asset_classes)

    def find_plate(self, vehicle_crop: np.ndarray) -> Optional[tuple[int, int, int, int]]:
        """
        Find a license plate inside a vehicle crop.
        Returns (x1,y1,x2,y2) in CROP coordinates, or None.
        Uses YOLO plate model when available, otherwise uses contour-based detection.
        """
        if vehicle_crop is None or vehicle_crop.size == 0:
            return None

        if self.plate_model is not None:
            try:
                results = self.plate_model(vehicle_crop, conf=self.plate_conf, verbose=False)
                if results and len(results[0].boxes) > 0:
                    box = results[0].boxes[0]
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    return (x1, y1, x2, y2)
            except Exception as e:
                log.debug("plate model error: %s", e)

        return self._find_plate_contour(vehicle_crop)

    @staticmethod
    def _find_plate_contour(crop: np.ndarray) -> tuple[int, int, int, int]:
        """
        Contour-based plate localisation — finds the most plate-like rectangle
        in the bottom 65% of the vehicle crop.

        Works by:
          1. Enhancing contrast with CLAHE
          2. Detecting edges with Canny
          3. Horizontally dilating to merge character blobs into one plate blob
          4. Filtering contours by plate aspect ratio (2:1 – 6:1)
        """
        import cv2 as _cv2
        import numpy as _np

        h, w = crop.shape[:2]

        # Plates are always in the lower portion of the vehicle
        y_start = int(h * 0.35)
        roi = crop[y_start:, :]
        rh, rw = roi.shape[:2]

        gray = _cv2.cvtColor(roi, _cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi.copy()

        # CLAHE boosts contrast on dirty / low-light plates
        clahe = _cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray  = clahe.apply(gray)

        # Gaussian blur before Canny reduces noise-induced false edges
        blurred = _cv2.GaussianBlur(gray, (5, 5), 0)
        edges   = _cv2.Canny(blurred, 40, 120)

        # Horizontal morphological close merges character edges into one plate blob
        kernel = _cv2.getStructuringElement(_cv2.MORPH_RECT, (22, 5))
        closed = _cv2.morphologyEx(edges, _cv2.MORPH_CLOSE, kernel)

        contours, _ = _cv2.findContours(closed, _cv2.RETR_EXTERNAL,
                                        _cv2.CHAIN_APPROX_SIMPLE)

        best_box   = None
        best_score = 0.0
        min_area   = rw * rh * 0.008   # at least 0.8% of search area

        for c in contours:
            x, y, cw, ch = _cv2.boundingRect(c)
            if cw * ch < min_area:
                continue
            aspect = cw / max(ch, 1)
            if not (2.0 <= aspect <= 7.0):
                continue
            # Score: prefer wider plates centred horizontally
            cx_offset = abs((x + cw / 2) - rw / 2) / rw   # 0=centre, 1=edge
            score = (cw * ch) * max(0.2, 1.0 - cx_offset)
            if score > best_score:
                best_score = score
                pad = 5
                best_box = (
                    max(0,  x  - pad),
                    y_start + max(0,  y  - pad),
                    min(w,  x  + cw + pad),
                    y_start + min(rh, y  + ch + pad),
                )

        if best_box:
            return best_box
        # Fallback: centre-bottom strip — tighter than the old 40% to reduce noise
        return (int(w * 0.15), int(h * 0.70), int(w * 0.85), int(h * 0.97))

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _yolo_to_detections(self, yolo_result, filter_classes: set) -> List[Detection]:
        names = yolo_result.names                  # {class_idx: name}
        boxes = yolo_result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        out: list[Detection] = []
        ids = boxes.id.int().tolist() if boxes.id is not None else [None] * len(boxes)
        for i in range(len(boxes)):
            cls = int(boxes.cls[i].item())
            name = str(names[cls]).lower()
            if filter_classes and name not in filter_classes:
                continue
            conf = float(boxes.conf[i].item())
            x1, y1, x2, y2 = map(int, boxes.xyxy[i].tolist())
            out.append(Detection(
                cls_name=name,
                confidence=conf,
                bbox=(x1, y1, x2, y2),
                track_id=ids[i] if i < len(ids) else None,
            ))
        return out
