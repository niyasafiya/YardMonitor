"""
scripts/export_to_onnx.py — Export YOLO models to ONNX for faster inference.

ONNX Runtime gives 2-3x speedup over PyTorch .pt on your MX330 because it
runs a graph-optimised, pre-compiled computation graph instead of dynamically
dispatching through Python/autograd.

── Install the right onnxruntime before running ───────────────────────────────
  CUDA 11.x (your MX330):   pip install onnxruntime-gpu==1.16.3
  CUDA 12.x:                pip install onnxruntime-gpu>=1.17.0
  CPU only:                 pip install onnxruntime>=1.16.0

── Run once, then switch config.yaml ──────────────────────────────────────────
  python scripts/export_to_onnx.py

  Then edit config.yaml:
      models:
        vehicle_detector: "models/yolov8n.onnx"
        plate_detector:   "models/license_plate.onnx"

── Flags ──────────────────────────────────────────────────────────────────────
  --vehicle  models/yolov8n.pt          source vehicle detector
  --plate    models/license_plate.pt    source plate detector
  --imgsz    640                        inference resolution (must match training)
  --opset    12                         ONNX opset (12 = widest onnxruntime compat)
  --device   0 | cpu                    export device (0 = first GPU)
  --no-simplify                         skip onnx-simplifier pass
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT      = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"


def _check_ort():
    """Warn if onnxruntime is not installed or is the wrong GPU variant."""
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        has_cuda = "CUDAExecutionProvider" in providers
        print(f"  onnxruntime {ort.__version__}  —  CUDA provider: {'yes' if has_cuda else 'NO (CPU only)'}")
        if not has_cuda:
            print(
                "\n  WARNING: GPU provider not available.\n"
                "  For CUDA 11.8 install: pip install onnxruntime-gpu==1.16.3\n"
            )
    except ImportError:
        print(
            "\n  WARNING: onnxruntime not installed.\n"
            "  For CUDA 11.8 install: pip install onnxruntime-gpu==1.16.3\n"
            "  For CPU only install:  pip install onnxruntime\n"
        )


def export_model(
    pt_path: Path,
    dest_dir: Path,
    imgsz: int,
    opset: int,
    device: str,
    simplify: bool,
) -> Path:
    from ultralytics import YOLO

    print(f"\n  Exporting {pt_path.name}  (imgsz={imgsz}, opset={opset}, device={device})")
    model = YOLO(str(pt_path))

    result = model.export(
        format="onnx",
        imgsz=imgsz,
        opset=opset,
        simplify=simplify,
        dynamic=False,   # fixed batch=1 is faster and avoids shape inference issues
        device=device,
    )

    exported = Path(result)
    dest = dest_dir / (pt_path.stem + ".onnx")
    if exported.resolve() != dest.resolve():
        shutil.move(str(exported), str(dest))
    print(f"  Saved → {dest}  ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


def main():
    ap = argparse.ArgumentParser(
        description="Export YOLO models to ONNX for ONNX Runtime inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--vehicle",    default=str(MODEL_DIR / "yolov8n.pt"))
    ap.add_argument("--plate",      default=str(MODEL_DIR / "license_plate.pt"))
    ap.add_argument("--imgsz",      default=640, type=int)
    ap.add_argument("--opset",      default=12,  type=int)
    ap.add_argument("--device",     default="0")
    ap.add_argument("--no-simplify", action="store_true")
    args = ap.parse_args()

    print("Checking ONNX Runtime installation…")
    _check_ort()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    exported = []

    for attr, label in [("vehicle", "vehicle detector"), ("plate", "plate detector")]:
        pt = Path(getattr(args, attr))
        if not pt.exists():
            print(f"\n  SKIP {label}: {pt} not found")
            continue
        try:
            out = export_model(
                pt, MODEL_DIR,
                imgsz=args.imgsz,
                opset=args.opset,
                device=args.device,
                simplify=not args.no_simplify,
            )
            exported.append(out)
        except Exception as e:
            print(f"\n  ERROR exporting {label}: {e}")
            print("  Try: --device cpu  if GPU export fails")
            sys.exit(1)

    print("\n── Done ─────────────────────────────────────────────────────────────")
    for p in exported:
        print(f"  {p}")

    print("""
Update config.yaml to switch to ONNX:

    models:
      vehicle_detector: "models/yolov8n.onnx"
      plate_detector:   "models/license_plate.onnx"

Then restart the server.  Expect 2-3x faster inference per frame.
""")


if __name__ == "__main__":
    main()
