"""
Generate a small synthetic 'gate' video so the system has something to chew on
for a demo even before you wire up real cameras.

It just draws a moving rectangle labelled with a fake plate so you can see the
pipeline running end-to-end on first launch.

NOTE: The synthetic video will NOT trigger real plate detection — YOLO is
trained on real vehicles, not coloured rectangles. Use this only to confirm the
*pipeline plumbing* is working. For an actual LPR demo, drop a real video at
data/sample_gate.mp4.

Run:
    python scripts/make_sample_video.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "sample_gate.mp4"
OUT.parent.mkdir(parents=True, exist_ok=True)

W, H, FPS, SECS = 1280, 720, 24, 10


def main():
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(OUT), fourcc, FPS, (W, H))
    if not vw.isOpened():
        print("Could not open video writer. Is ffmpeg installed?")
        sys.exit(1)

    for f in range(FPS * SECS):
        img = np.full((H, W, 3), 30, dtype=np.uint8)
        # ground gradient
        for y in range(H // 2, H):
            shade = int(60 + (y - H/2) / (H/2) * 40)
            img[y, :] = (shade, shade, shade)
        # direction line guide
        cv2.line(img, (0, int(H*0.55)), (W, int(H*0.55)), (60, 60, 80), 1)

        # moving rectangle = "vehicle"
        progress = f / (FPS * SECS)
        cy = int(120 + progress * (H - 240))
        cx = W // 2
        cv2.rectangle(img, (cx - 140, cy - 60), (cx + 140, cy + 60), (40, 80, 160), -1)
        cv2.rectangle(img, (cx - 80, cy + 20), (cx + 80, cy + 50), (240, 240, 240), -1)
        cv2.putText(img, "KL07BX1234", (cx - 75, cy + 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (10, 10, 10), 2)
        cv2.putText(img, f"frame {f}", (16, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        vw.write(img)
    vw.release()
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
