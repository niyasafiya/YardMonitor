"""
Download the YOLO models required by Yard Monitor.

* yolov8n.pt        — vehicle detector (auto-downloaded by ultralytics on first use,
                      but we pre-fetch it here so the first run isn't blocked).
* license_plate.pt  — license plate detector. We try a couple of well-known
                      public sources; if all fail you'll be told to drop your
                      own model in models/license_plate.pt.

Run once:

    python scripts/download_models.py
"""
from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# YOLOv8n (vehicle detector) — pulled by ultralytics from its release CDN.
VEHICLE_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt"
VEHICLE_OUT = MODEL_DIR / "yolov8n.pt"

# Public license-plate YOLO weights. Several open-source ones exist; the
# first URL that works wins. If none work, the system still runs with a
# fallback (crop the lower half of the vehicle bbox + OCR).
PLATE_CANDIDATES = [
    # Try this first — keremberke's lpr model converted to .pt
    "https://huggingface.co/keremberke/yolov8n-license-plate/resolve/main/best.pt",
]
PLATE_OUT = MODEL_DIR / "license_plate.pt"


def download(url: str, dest: Path) -> bool:
    print(f"  → {url}")
    try:
        with urllib.request.urlopen(url, timeout=60) as resp, open(dest, "wb") as f:
            chunk = resp.read(1 << 16)
            total = 0
            while chunk:
                f.write(chunk)
                total += len(chunk)
                chunk = resp.read(1 << 16)
        print(f"  OK {dest.name}  ({total/1e6:.1f} MB)")
        return True
    except Exception as e:
        print(f"  ✗ failed: {e}")
        if dest.exists():
            dest.unlink(missing_ok=True)
        return False


def main():
    print("Downloading models into", MODEL_DIR)

    if VEHICLE_OUT.exists():
        print(f"yolov8n.pt already present, skipping.")
    else:
        print("\n[1/2] Vehicle detector (yolov8n.pt):")
        if not download(VEHICLE_URL, VEHICLE_OUT):
            print("  Could not fetch yolov8n.pt. ultralytics will try again on first run.")

    if PLATE_OUT.exists():
        print(f"\nlicense_plate.pt already present, skipping.")
    else:
        print("\n[2/2] License-plate detector (license_plate.pt):")
        success = False
        for url in PLATE_CANDIDATES:
            if download(url, PLATE_OUT):
                success = True
                break
        if not success:
            print("\n  ⚠  Could not fetch a plate detector. The system will still")
            print("     work using a heuristic crop + OCR; accuracy is lower.")
            print("     To improve accuracy:")
            print("       1. Train your own with `yolo detect train` on a plate dataset")
            print("          (Roboflow has free ones).")
            print("       2. Or drop any YOLOv5/v8 plate weights at models/license_plate.pt")

    print("\nDone.")


if __name__ == "__main__":
    main()
