"""
scripts/train_plate_detector.py

Train a YOLOv8 license-plate detector and save it to models/license_plate.pt.

--------------------------------------------------------------------
QUICK START — no account needed (~500 MB download, ~2 500 images):

    pip install datasets
    python scripts/train_plate_detector.py

Downloads keremberke/license-plate-object-detection from HuggingFace
and trains YOLOv8s for 100 epochs on your GPU.

--------------------------------------------------------------------
BEST FOR INDIAN PLATES — free Roboflow account (2-min sign-up):

    pip install roboflow
    python scripts/train_plate_detector.py --source roboflow \\
        --rf-key  YOUR_FREE_API_KEY \\
        --rf-workspace  <workspace-slug> \\
        --rf-project    <project-slug> \\
        --rf-version    1

Recommended Indian LP datasets at https://universe.roboflow.com:
  • "indian-license-plates"               (search term)
  • "vehicle-registration-plate-india"    (search term)
  • "license-plate-recognition-rxg4e"     (mixed, large)

--------------------------------------------------------------------
DIRECT ZIP — dataset already exported to YOLO format:

    python scripts/train_plate_detector.py --source zip \\
        --zip-url https://example.com/plates.zip

    ZIP must contain images/train, labels/train, images/valid,
    labels/valid sub-folders and a data.yaml.

--------------------------------------------------------------------
LOCAL — dataset already on disk:

    python scripts/train_plate_detector.py --source local \\
        --data-yaml /path/to/data.yaml

--------------------------------------------------------------------
TRAINING FLAGS:
  --model    yolov8n/s/m/l/x   (default: yolov8s)
  --epochs   N                 (default: 100)
  --imgsz    N                 (default: 640)
  --batch    N                 (default: 16)
  --workers  N                 (default: 4)
  --device   0 / cpu           (default: 0 — first CUDA GPU)
  --name     run-name          (default: plate_train)
  --no-copy                    skip copying result to models/license_plate.pt
"""
from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT      = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
DATA_DIR  = ROOT / "data" / "plate_dataset"

# ---------------------------------------------------------------------------
# Dataset downloaders
# ---------------------------------------------------------------------------

def _save_yolo_label(label_path: Path, bboxes: list, img_w: int, img_h: int):
    """Write a YOLO .txt label file from a list of COCO-format bboxes."""
    with open(label_path, "w") as f:
        for bbox in bboxes:
            x_min, y_min, bw, bh = bbox
            cx  = (x_min + bw / 2) / img_w
            cy  = (y_min + bh / 2) / img_h
            nw  = bw / img_w
            nh  = bh / img_h
            # Clamp to valid range
            cx  = max(0.0, min(1.0, cx))
            cy  = max(0.0, min(1.0, cy))
            nw  = max(0.0, min(1.0, nw))
            nh  = max(0.0, min(1.0, nh))
            if nw > 0 and nh > 0:
                f.write(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")


def _write_data_yaml(dest_dir: Path) -> Path:
    yaml_path = dest_dir / "data.yaml"
    yaml_path.write_text(
        f"path: {dest_dir.resolve()}\n"
        "train: images/train\n"
        "val:   images/valid\n"
        "nc: 1\n"
        "names:\n"
        "  0: license_plate\n"
    )
    return yaml_path


def _coco_zip_to_yolo(zip_path: str, img_dir: Path, label_dir: Path):
    """Extract a COCO-annotated ZIP (images + _annotations.coco.json) to YOLO layout."""
    import json

    img_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()

        # Load COCO annotations
        coco_names = [n for n in names if n.endswith("_annotations.coco.json")]
        if not coco_names:
            raise RuntimeError(f"No _annotations.coco.json found in {zip_path}")
        with zf.open(coco_names[0]) as f:
            coco = json.load(f)

        # Build lookup: image_id → {file_name, width, height}
        img_meta = {img["id"]: img for img in coco["images"]}

        # Build lookup: image_id → list of COCO bboxes
        ann_map: dict[int, list] = {}
        for ann in coco["annotations"]:
            ann_map.setdefault(ann["image_id"], []).append(ann["bbox"])

        # Extract images and write YOLO labels
        img_members = {n for n in names if n.lower().endswith((".jpg", ".jpeg", ".png"))}
        for img_info in coco["images"]:
            fname = img_info["file_name"]
            if fname not in img_members:
                continue
            stem  = Path(fname).stem
            bboxes = ann_map.get(img_info["id"], [])
            with zf.open(fname) as src:
                (img_dir / fname).write_bytes(src.read())
            _save_yolo_label(
                label_dir / f"{stem}.txt",
                bboxes,
                img_info["width"],
                img_info["height"],
            )

    return len(coco["images"])


def download_huggingface(dest_dir: Path) -> Path:
    """Download keremberke/license-plate-object-detection ZIP files directly via huggingface_hub."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        sys.exit(
            "\nMissing package: huggingface_hub\n"
            "Install it with:  pip install huggingface_hub\n"
        )

    repo_id = "keremberke/license-plate-object-detection"
    zips = {
        "train": "data/train.zip",
        "valid": "data/valid.zip",
    }

    os_env = {"HF_HUB_DISABLE_SYMLINKS_WARNING": "1"}
    import os
    for k, v in os_env.items():
        os.environ[k] = v

    total_imgs = 0
    for split, hf_path in zips.items():
        print(f"  Downloading {hf_path} from HuggingFace…")
        local_zip = hf_hub_download(repo_id, hf_path, repo_type="dataset")
        print(f"  Extracting {split} split…")
        n = _coco_zip_to_yolo(
            local_zip,
            dest_dir / "images" / split,
            dest_dir / "labels" / split,
        )
        print(f"  {split}: {n} images")
        total_imgs += n

    print(f"\n  Done — {total_imgs} total images in {dest_dir}\n")
    return _write_data_yaml(dest_dir)


def download_roboflow(
    dest_dir: Path,
    api_key: str,
    workspace: str,
    project: str,
    version: int,
) -> Path:
    """Download a Roboflow dataset in YOLOv8 format."""
    try:
        from roboflow import Roboflow
    except ImportError:
        sys.exit(
            "\nMissing package: roboflow\n"
            "Install it with:  pip install roboflow\n"
        )

    print(f"Downloading {workspace}/{project} v{version} from Roboflow…")
    rf      = Roboflow(api_key=api_key)
    dataset = rf.workspace(workspace).project(project).version(version).download(
        "yolov8", location=str(dest_dir)
    )

    # Roboflow may write data.yaml inside a sub-folder
    yaml_path = dest_dir / "data.yaml"
    if not yaml_path.exists():
        candidates = sorted(dest_dir.rglob("data.yaml"), key=lambda p: len(p.parts))
        if candidates:
            yaml_path = candidates[0]

    if not yaml_path.exists():
        sys.exit(f"ERROR: data.yaml not found under {dest_dir}")

    print(f"  Done — dataset at {yaml_path}\n")
    return yaml_path


def download_zip(dest_dir: Path, url: str) -> Path:
    """Download and extract a YOLO-format dataset ZIP."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / "_dataset.zip"

    print(f"Downloading {url} …")
    with urllib.request.urlopen(url, timeout=120) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        done  = 0
        with open(zip_path, "wb") as f:
            while True:
                chunk = resp.read(1 << 16)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  {done / total * 100:.1f}%", end="", flush=True)
    print(f"\r  {done / 1e6:.1f} MB downloaded")

    print("  Extracting…")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    zip_path.unlink(missing_ok=True)

    candidates = sorted(dest_dir.rglob("data.yaml"), key=lambda p: len(p.parts))
    if not candidates:
        sys.exit(
            f"ERROR: No data.yaml found in the extracted ZIP ({dest_dir}).\n"
            "Make sure the ZIP is in YOLO format with images/train, labels/train, "
            "images/valid, labels/valid, and data.yaml."
        )
    print(f"  Done — dataset at {candidates[0]}\n")
    return candidates[0]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(data_yaml: Path, args: argparse.Namespace):
    from ultralytics import YOLO

    print(f"Loading base model: {args.model}.pt …")
    model = YOLO(f"{args.model}.pt")

    print(f"Training on {data_yaml}  [epochs={args.epochs}, imgsz={args.imgsz}, "
          f"batch={args.batch}, device={args.device}]\n")

    model.train(
        data=str(data_yaml.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        name=args.name,
        project=str(ROOT / "runs" / "plate_train"),
        exist_ok=True,
        # Augmentation tuned for plates
        flipud=0.0,       # plates don't appear upside down
        fliplr=0.5,
        degrees=5.0,      # slight rotation tolerance
        scale=0.5,        # zoom augmentation
        mosaic=0.5,
        close_mosaic=10,  # disable mosaic in final 10 epochs for stability
    )

    # The run dir ultralytics creates
    run_dir = ROOT / "runs" / "plate_train" / args.name
    best_pt = run_dir / "weights" / "best.pt"

    if not best_pt.exists():
        # Fallback: find the most-recently-modified best.pt
        candidates = sorted(
            (ROOT / "runs" / "plate_train").rglob("best.pt"),
            key=lambda p: p.stat().st_mtime,
        )
        if candidates:
            best_pt = candidates[-1]
        else:
            print("\nWARNING: Could not locate best.pt — check runs/plate_train/")
            return None

    print(f"\nBest weights: {best_pt}")
    return best_pt


def copy_to_models(best_pt: Path, no_copy: bool):
    dest = MODEL_DIR / "license_plate.pt"
    if no_copy:
        print(f"\nSkipping copy (--no-copy set).")
        print(f"Trained weights at: {best_pt}")
        return

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    backup = MODEL_DIR / "license_plate_prev.pt"
    if dest.exists():
        shutil.copy2(dest, backup)
        print(f"  Previous model backed up → {backup.name}")

    shutil.copy2(best_pt, dest)
    print(f"\nModel saved to {dest}")
    print("The yard monitor will use the new model on next restart.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train YOLOv8 license-plate detector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument("--source", default="huggingface",
                   choices=["huggingface", "roboflow", "zip", "local"],
                   help="Dataset source (default: huggingface)")
    p.add_argument("--dest-dir", default=str(DATA_DIR),
                   help="Where to save the downloaded dataset")

    # Roboflow options
    rf = p.add_argument_group("Roboflow options (--source roboflow)")
    rf.add_argument("--rf-key",       default="", help="Roboflow API key")
    rf.add_argument("--rf-workspace", default="", help="Workspace slug")
    rf.add_argument("--rf-project",   default="", help="Project slug")
    rf.add_argument("--rf-version",   default=1,  type=int, help="Dataset version")

    # ZIP option
    p.add_argument("--zip-url",  default="", help="Direct URL of a YOLO-format ZIP")

    # Local option
    p.add_argument("--data-yaml", default="", help="Path to existing data.yaml")

    # Training hyper-parameters
    p.add_argument("--model",   default="yolov8s",
                   help="Base YOLO model (yolov8n/s/m/l/x — default: yolov8s)")
    p.add_argument("--epochs",  default=100, type=int)
    p.add_argument("--imgsz",   default=640, type=int)
    p.add_argument("--batch",   default=16,  type=int)
    p.add_argument("--workers", default=4,   type=int)
    p.add_argument("--device",  default="0",
                   help="CUDA device index or 'cpu' (default: 0)")
    p.add_argument("--name",    default="plate_train",
                   help="Run name under runs/plate_train/")
    p.add_argument("--no-copy", action="store_true",
                   help="Do not copy best.pt to models/license_plate.pt")

    return p.parse_args()


def main():
    args = parse_args()
    dest_dir = Path(args.dest_dir)

    # ---- Obtain data.yaml ----
    if args.source == "huggingface":
        dest_dir.mkdir(parents=True, exist_ok=True)
        yaml_path = download_huggingface(dest_dir)

    elif args.source == "roboflow":
        if not args.rf_key:
            sys.exit("ERROR: --rf-key is required for --source roboflow")
        if not args.rf_workspace or not args.rf_project:
            sys.exit("ERROR: --rf-workspace and --rf-project are required")
        dest_dir.mkdir(parents=True, exist_ok=True)
        yaml_path = download_roboflow(
            dest_dir, args.rf_key, args.rf_workspace,
            args.rf_project, args.rf_version,
        )

    elif args.source == "zip":
        if not args.zip_url:
            sys.exit("ERROR: --zip-url is required for --source zip")
        yaml_path = download_zip(dest_dir, args.zip_url)

    elif args.source == "local":
        if not args.data_yaml:
            sys.exit("ERROR: --data-yaml is required for --source local")
        yaml_path = Path(args.data_yaml)
        if not yaml_path.exists():
            sys.exit(f"ERROR: {yaml_path} does not exist")

    # ---- Train ----
    best_pt = train(yaml_path, args)
    if best_pt is None:
        sys.exit(1)

    # ---- Deploy ----
    copy_to_models(best_pt, args.no_copy)

    print("\nDone! Restart the yard monitor to use the new model.")
    print("   python main.py   (or restart start.bat / start.sh)")


if __name__ == "__main__":
    main()
