"""
Material / cargo inspection service.

Looks at an image or short clip of what a vehicle (or asset) is carrying,
detects the items inside it with the shared YOLOv8 object model, and maps each
detected object class to a "material" name. The router then compares those
materials against an editable allow-list to GRANT or DENY yard entry.

Mirrors services.ppe_service: it loads its own ultralytics YOLO model (all COCO
classes, unlike yolo_service which filters to vehicles only) and exposes a small
analyze_frame() + draw_boxes() surface.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# COCO class -> friendly material name.
# The object model is general-purpose (COCO 80), so we translate the classes
# that plausibly appear as cargo into material names. Anything not in this map
# falls through using its raw class name, so the allow-list can still cover it.
# person + vehicle classes are treated as the *carrier*, never as cargo.
# ---------------------------------------------------------------------------
_CARRIER_CLASSES = {
    "person", "car", "truck", "bus", "motorcycle", "bicycle", "train",
    "airplane", "boat",
}

_MATERIAL_MAP: Dict[str, str] = {
    "bottle":      "bottle",
    "wine glass":  "bottle",
    "cup":         "bottle",
    "vase":        "barrel",
    "suitcase":    "suitcase",
    "handbag":     "backpack",
    "backpack":    "backpack",
    "book":        "book",
    "laptop":      "electronics",
    "tv":          "electronics",
    "keyboard":    "electronics",
    "microwave":   "electronics",
    "refrigerator": "appliance",
    "oven":        "appliance",
    "chair":       "furniture",
    "couch":       "furniture",
    "bench":       "furniture",
    "potted plant": "plant",
    "sports ball": "sack",
}

_COLOR_ALLOW = (40, 201, 168)    # green-teal for allowed material
_COLOR_DENY = (110, 86, 242)     # red for disallowed material (BGR)


def normalise(name: str) -> str:
    """Lowercase, single-spaced material key used everywhere (DB, compare)."""
    return " ".join((name or "").strip().lower().split())


def class_to_material(cls_name: str) -> str:
    key = normalise(cls_name)
    return _MATERIAL_MAP.get(key, key)


# ---------------------------------------------------------------------------
# Detector (own YOLO instance — all classes, no vehicle filter)
# ---------------------------------------------------------------------------
class MaterialDetector:
    CONF = 0.35

    def __init__(self):
        self._model = None
        self._names: Dict[int, str] = {}
        self._load()

    def _load(self):
        """Load the object model, preferring a larger (more accurate) one.

        Order: $SENTINEL_OBJECT_MODEL → yolov8m (better recall on big/odd-angle
        items like a bed on a truck) → yolov8n (always shipped as a fallback).
        The first candidate that loads wins; if the bigger weights are missing
        and cannot be fetched, we quietly fall back so detection still works.
        """
        try:
            from ultralytics import YOLO
        except Exception as exc:                        # pragma: no cover
            log.warning("[Material] ultralytics unavailable (%s) — detection disabled", exc)
            self._model = None
            return

        candidates = []
        env = os.environ.get("SENTINEL_OBJECT_MODEL")
        if env:
            candidates.append(env)
        candidates += ["models/yolov8m.pt", "yolov8m.pt",
                       "models/yolov8n.pt", "yolov8n.pt"]

        for cand in candidates:
            # For a plain filename (no dir) ultralytics will auto-download it;
            # for an explicit path, only try it when the file is actually there.
            if ("/" in cand or "\\" in cand) and not Path(cand).exists():
                continue
            try:
                self._model = YOLO(cand)
                self._names = self._model.names
                log.info("[Material] Loaded object model %s", cand)
                return
            except Exception as exc:                    # pragma: no cover
                log.warning("[Material] could not load %s (%s)", cand, exc)
        log.warning("[Material] no object model could be loaded — detection disabled")
        self._model = None

    @property
    def available(self) -> bool:
        return self._model is not None

    def analyze_frame(self, frame: np.ndarray, conf: float | None = None,
                      imgsz: int = 640) -> dict:
        """
        Detect cargo items in one BGR frame.

        Returns { boxes:[{xyxy, cls, material, conf}], materials:{name:count} }.
        Carrier classes (person/vehicle) are excluded from the cargo list.

        conf / imgsz can be overridden by callers that need higher recall (e.g.
        asset tracking uses a lower confidence and larger image size so oddly
        posed items — a bed strapped on a truck — are still picked up).
        """
        if self._model is None:
            raise RuntimeError("Material detection model unavailable")

        results = self._model(frame, conf=(self.CONF if conf is None else conf),
                              verbose=False, imgsz=imgsz)
        names = results[0].names or self._names
        boxes = results[0].boxes

        out_boxes: List[dict] = []
        materials: Dict[str, int] = {}
        if boxes is not None:
            for i in range(len(boxes)):
                cls_name = normalise(str(names[int(boxes.cls[i].item())]))
                if cls_name in _CARRIER_CLASSES:
                    continue
                material = class_to_material(cls_name)
                x1, y1, x2, y2 = map(int, boxes.xyxy[i].tolist())
                out_boxes.append({
                    "xyxy": (x1, y1, x2, y2),
                    "cls": cls_name,
                    "material": material,
                    "conf": round(float(boxes.conf[i].item()), 3),
                })
                materials[material] = materials.get(material, 0) + 1
        return {"boxes": out_boxes, "materials": materials}


def draw_boxes(frame: np.ndarray, boxes: List[dict], allowed: set) -> np.ndarray:
    """Annotate the frame — green box for allowed material, red for disallowed."""
    out = frame.copy()
    for b in boxes:
        x1, y1, x2, y2 = b["xyxy"]
        ok = b["material"] in allowed
        color = _COLOR_ALLOW if ok else _COLOR_DENY
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        tag = "OK" if ok else "NOT ALLOWED"
        label = f"{b['material']} {b['conf']:.2f} [{tag}]"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(out, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (12, 12, 12), 1)
    return out


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_detector: Optional[MaterialDetector] = None


def get_material_detector() -> MaterialDetector:
    global _detector
    if _detector is None:
        _detector = MaterialDetector()
    return _detector
