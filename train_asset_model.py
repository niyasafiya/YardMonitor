"""
Train a custom LOAD / ASSET classifier for Sentinel asset tracking.

Off-the-shelf detectors can't recognise bulk or wrapped cargo (e.g. a truck
stacked with shrink-wrapped mattresses). This script fine-tunes a small model on
YOUR OWN example photos so the console can recognise the load types you actually
handle.

--------------------------------------------------------------------------
HOW TO USE
--------------------------------------------------------------------------
1. Sort example photos into one folder per load type:

       data/asset_images/mattress/*.jpg
       data/asset_images/box/*.jpg
       data/asset_images/barrel/*.jpg
       data/asset_images/<your-load-type>/*.jpg

   - Aim for ~30-100 varied photos per type (different trucks, angles, light).
   - Crop loosely to the load if you can; whole-truck photos work too.
   - You need at least 2 non-empty folders.

2. Run:

       python train_asset_model.py                 # ~30 epochs (default)
       ASSET_EPOCHS=50 python train_asset_model.py  # more epochs (PowerShell: set env first)

3. When it finishes it writes  models/asset_cls.pt  and the backend picks it up
   automatically on the next restart. Register an asset whose *category* matches
   a trained class name (e.g. category "mattress") so detections match a tag.
--------------------------------------------------------------------------
"""
from __future__ import annotations

import os
import random
import shutil
import sys
from pathlib import Path

SRC = Path("data/asset_images")          # you drop photos here, one folder per class
DS = Path("data/asset_dataset")          # auto-built train/val split (safe to delete)
OUT = Path("models/asset_cls.pt")        # trained model the backend loads
VAL_FRAC = 0.2
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
EPOCHS = int(os.environ.get("ASSET_EPOCHS", "30"))
IMGSZ = int(os.environ.get("ASSET_IMGSZ", "224"))


def build_split() -> dict:
    """Copy data/asset_images/<class>/* into an 80/20 train/val dataset."""
    classes = [d for d in sorted(SRC.iterdir()) if d.is_dir()]
    counts: dict = {}
    usable = []
    for c in classes:
        imgs = [p for p in c.iterdir() if p.suffix.lower() in IMG_EXT]
        if imgs:
            usable.append((c, imgs))
    if len(usable) < 2:
        sys.exit(
            f"Need at least 2 non-empty class folders under {SRC}/.\n"
            f"Add photos, e.g. {SRC}/mattress/*.jpg and {SRC}/box/*.jpg, then re-run."
        )

    if DS.exists():
        shutil.rmtree(DS)
    for c, imgs in usable:
        random.shuffle(imgs)
        nval = int(len(imgs) * VAL_FRAC)
        if len(imgs) > 1:
            nval = max(1, nval)
        val, train = imgs[:nval], imgs[nval:]
        if not train:                     # single image → use it for both
            train = val
        for split, group in (("train", train), ("val", val or train)):
            dst = DS / split / c.name
            dst.mkdir(parents=True, exist_ok=True)
            for p in group:
                shutil.copy(p, dst / p.name)
        counts[c.name] = (len(train), len(val or train))
    return counts


def main():
    counts = build_split()
    print("\nDataset built:")
    for k, (tr, va) in counts.items():
        print(f"  {k:18s} train={tr:4d}  val={va:4d}")
    total = sum(tr + va for tr, va in counts.values())
    if total < 20:
        print("\n[!] Very few images - accuracy will be poor. Aim for ~30+ per class.")

    from ultralytics import YOLO
    print(f"\nTraining yolov8n-cls for {EPOCHS} epochs (imgsz={IMGSZ})…")
    model = YOLO("yolov8n-cls.pt")        # small classifier, transfer-learned
    model.train(data=str(DS.resolve()), epochs=EPOCHS, imgsz=IMGSZ, verbose=True)

    best = Path(model.trainer.best)       # runs/classify/trainX/weights/best.pt
    OUT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best, OUT)
    print(f"\n[OK] Saved custom load classifier -> {OUT}")
    print("  Restart the backend (python main.py) to activate it.")
    print("  Classes:", ", ".join(counts.keys()))


if __name__ == "__main__":
    main()
