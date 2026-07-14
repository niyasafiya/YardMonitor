"""
Asset detection service — recognise the assets a truck is carrying.

Reuses the shared YOLOv8 object model already loaded by services.material_service
(so no second model is loaded into memory), and re-maps each detected object
class to a friendly *asset category* (case, bag, container, electronics,
appliance, furniture…). The router then matches those categories against a
register of known company assets to surface each item's real asset tag / name.

Exposes analyze_frame() + draw_boxes(), mirroring material_service so the
asset router can follow the same upload / live-frame flow as the others.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from services import material_service as ms

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# The object model is general-purpose (COCO 80). For asset tracking we report
# the *specific* class the model recognises (e.g. "bed", "laptop", "suitcase")
# as the asset's category, rather than collapsing it into a broad group — an
# operator wants to know it is a bed, not merely "furniture". A couple of raw
# COCO names are tidied for display; everything else is used verbatim so the
# register can reference the exact item. person + vehicle classes are the
# *carrier* (the truck itself), never an asset.
# ---------------------------------------------------------------------------
_CARRIER_CLASSES = {
    "person", "car", "truck", "bus", "motorcycle", "bicycle", "train",
    "airplane", "boat",
}

# Light cosmetic renames only — these keep the specific item, just friendlier.
_ASSET_RENAME: Dict[str, str] = {
    "cell phone":   "phone",
    "potted plant": "plant",
    "sports ball":  "ball",
    "tv":           "tv monitor",
}

_COLOR_KNOWN = (40, 201, 168)     # green-teal — matched to a registered asset (BGR)
_COLOR_GENERIC = (235, 170, 60)   # amber — detected asset, not in the register


def normalise(name: str) -> str:
    """Lowercase, single-spaced key used everywhere (DB, compare)."""
    return " ".join((name or "").strip().lower().split())


def class_to_category(cls_name: str) -> str:
    """Recognised item name — the specific COCO class, lightly tidied."""
    key = normalise(cls_name)
    return _ASSET_RENAME.get(key, key)


# Asset detection favours recall over precision: a bed / crate / drum strapped
# on a truck is often at an odd angle and lower confidence than a catalogue
# photo, so we scan at a lower threshold and a larger image size than the
# material module. Both are env-tunable.
CONF = float(os.environ.get("ASSET_CONF", "0.20"))
IMGSZ = int(os.environ.get("ASSET_IMGSZ", "1280"))


def available() -> bool:
    """Asset detection rides on the material detector's YOLO model."""
    return ms.get_material_detector().available


def analyze_frame(frame: np.ndarray, conf: float | None = None,
                  imgsz: int | None = None) -> dict:
    """
    Detect assets in one BGR frame.

    Returns { boxes:[{xyxy, cls, category, conf}], categories:{name:count} }.
    Carrier classes (person / vehicle) are excluded.
    """
    det = ms.get_material_detector()
    if not det.available:
        raise RuntimeError("Asset detection model unavailable")

    raw = det.analyze_frame(frame, conf=(CONF if conf is None else conf),
                            imgsz=(IMGSZ if imgsz is None else imgsz))
    out_boxes: List[dict] = []
    categories: Dict[str, int] = {}
    for b in raw["boxes"]:
        cls_name = normalise(b["cls"])
        if cls_name in _CARRIER_CLASSES:
            continue
        category = class_to_category(cls_name)
        out_boxes.append({
            "xyxy": tuple(b["xyxy"]),
            "cls": cls_name,
            "category": category,
            "conf": b["conf"],
        })
        categories[category] = categories.get(category, 0) + 1
    return {"boxes": out_boxes, "categories": categories}


def draw_boxes(frame: np.ndarray, boxes: List[dict], label_by_cat: dict) -> np.ndarray:
    """
    Annotate the frame. `label_by_cat` maps a category -> registered asset label
    (e.g. "GEN-042 · Diesel Generator") so matched assets show their real tag in
    green; unmatched detections show the generic category in amber.
    """
    out = frame.copy()
    for b in boxes:
        x1, y1, x2, y2 = b["xyxy"]
        reg = label_by_cat.get(b["category"])
        color = _COLOR_KNOWN if reg else _COLOR_GENERIC
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{reg} {b['conf']:.2f}" if reg else f"{b['category']} {b['conf']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(out, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (12, 12, 12), 1)
    return out


def draw_load_banner(frame: np.ndarray, text: str, matched: bool) -> np.ndarray:
    """Draw a top banner for a whole-load classification (no bounding box)."""
    out = frame
    color = _COLOR_KNOWN if matched else _COLOR_GENERIC
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(out, (0, 0), (max(tw + 20, 220), th + 20), color, -1)
    cv2.putText(out, text, (10, th + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (12, 12, 12), 2)
    return out


# ---------------------------------------------------------------------------
# Custom LOAD classifier (optional) — recognises bulk/wrapped cargo the generic
# detector can't (e.g. mattresses). Trained via train_asset_model.py, which
# writes models/asset_cls.pt. When that file is absent this is simply disabled.
# ---------------------------------------------------------------------------
LOAD_MODEL_PATH = Path(os.environ.get("ASSET_LOAD_MODEL", "models/asset_cls.pt"))
LOAD_CONF = float(os.environ.get("ASSET_LOAD_CONF", "0.55"))

_load_model = None
_load_tried = False


def get_load_classifier():
    """Lazy-load the custom classifier, or None if it hasn't been trained yet."""
    global _load_model, _load_tried
    if not _load_tried:
        _load_tried = True
        if LOAD_MODEL_PATH.exists():
            try:
                from ultralytics import YOLO
                _load_model = YOLO(str(LOAD_MODEL_PATH))
                log.info("[Asset] Loaded custom load classifier %s (%d classes)",
                         LOAD_MODEL_PATH, len(_load_model.names))
            except Exception as exc:                    # pragma: no cover
                log.warning("[Asset] load classifier failed to load (%s)", exc)
                _load_model = None
        else:
            log.info("[Asset] No custom load classifier yet (train via train_asset_model.py)")
    return _load_model


def load_classifier_ready() -> bool:
    return get_load_classifier() is not None


def classify_load(frame: np.ndarray) -> Optional[dict]:
    """
    Classify the overall load with the custom model.

    Returns {label, conf, confident} or None when no model is trained.
    `confident` is True only when conf >= LOAD_CONF, so callers can distinguish
    a solid recognition from a weak best-guess.
    """
    m = get_load_classifier()
    if m is None:
        return None
    try:
        res = m(frame, verbose=False)[0]
        probs = res.probs
        if probs is None:
            return None
        top = int(probs.top1)
        conf = round(float(probs.top1conf), 3)
        return {"label": normalise(res.names[top]), "conf": conf,
                "confident": conf >= LOAD_CONF}
    except Exception:                                   # pragma: no cover
        log.exception("classify_load failed")
        return None
