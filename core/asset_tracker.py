"""
Asset tracker — keeps an inventory of objects currently in the yard.

Strategy: each tracked object (vehicle/container) gets a stable asset_code
of the form "<asset_type>-<track_id>". As long as the YOLO tracker keeps
the same track_id across frames, the asset stays mapped to it. When the
object is gone for `grace_sec` seconds, it's marked as not present.

For higher fidelity in production you'd link asset_code to a recognised
license plate where available. We do that opportunistically here.
"""
from __future__ import annotations

import logging
import time
from typing import Iterable, Optional

import cv2
import numpy as np

from . import database as db
from .detector import Detection

log = logging.getLogger(__name__)


class AssetTracker:
    def __init__(self, camera_id: str, snapshot_dir: str = "data/snapshots/assets",
                 grace_sec: int = 60):
        self.camera_id = camera_id
        self.snapshot_dir = snapshot_dir
        self.grace_sec = grace_sec
        self._last_seen: dict[str, float] = {}
        import os
        os.makedirs(snapshot_dir, exist_ok=True)

    # ------------------------------------------------------------------

    def update(self, frame: np.ndarray, detections: Iterable[Detection],
               plate_hint: Optional[str] = None) -> list[dict]:
        """
        Update inventory from a fresh frame of detections.
        Returns the list of currently-present assets (dicts).
        """
        seen_codes: set[str] = set()
        now = time.time()

        for det in detections:
            if det.track_id is None:
                continue
            code = f"{det.cls_name}-{det.track_id}"
            seen_codes.add(code)
            self._last_seen[code] = now

            bbox_s = ",".join(map(str, det.bbox))
            # Save a snapshot the first time we see this asset
            snapshot_path = self._save_snapshot_if_new(frame, det, code)

            db.upsert_asset(
                asset_code=code,
                asset_type=det.cls_name,
                plate=plate_hint,
                description=f"Auto-detected {det.cls_name} (track {det.track_id})",
                last_camera=self.camera_id,
                last_bbox=bbox_s,
            )
            if snapshot_path:
                db.audit("asset_seen", actor="system",
                         asset_code=code, camera=self.camera_id, bbox=bbox_s)

        # Mark anything we haven't seen for `grace_sec` as gone
        db.mark_assets_absent(self.camera_id, seen_codes, grace_sec=self.grace_sec)

        return db.list_assets(present_only=True)

    # ------------------------------------------------------------------

    def _save_snapshot_if_new(self, frame: np.ndarray, det: Detection,
                              code: str) -> Optional[str]:
        # Only save once per asset to avoid disk bloat.
        import os
        out = os.path.join(self.snapshot_dir, f"{code}.jpg")
        if os.path.exists(out):
            return None
        x1, y1, x2, y2 = det.bbox
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(frame.shape[1], x2); y2 = min(frame.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame[y1:y2, x1:x2]
        try:
            cv2.imwrite(out, crop)
            return out
        except Exception as e:
            log.debug("snapshot save failed: %s", e)
            return None
