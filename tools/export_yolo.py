from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    p = argparse.ArgumentParser(description="Export a fixed-size YOLO ONNX model for SpectraTrack")
    p.add_argument("--model", default="yolo11n.pt")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--output", default="models/yolo11n.onnx")
    p.add_argument("--name", default="yolo11n-coco")
    p.add_argument("--source", default="")
    p.add_argument("--android-copy", action="store_true", help="Also copy the model into Android assets")
    args = p.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Install exporter first: pip install ultralytics onnx onnxslim") from exc

    root = Path(__file__).resolve().parents[1]
    model = YOLO(args.model)
    exported = Path(model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=True,
        dynamic=False,
    ))

    out = (root / args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(exported, out)
    digest = sha256(out)

    source = args.source or f"Local Ultralytics export from {args.model}"
    manifest = {
        "name": args.name,
        "sha256": digest,
        "input_size": args.imgsz,
        "source": source,
        "format": "YOLO ONNX xywh+class-scores or compatible end-to-end xyxy+score+class",
        "notes": f"opset={args.opset}; dynamic=false; simplify=true",
    }
    manifest_path = out.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"PC model:      {out}")
    print(f"Manifest:      {manifest_path}")
    print(f"SHA-256:       {digest}")

    if args.android_copy:
        android_out = root / "android" / "app" / "src" / "main" / "assets" / out.name
        android_out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out, android_out)
        print(f"Android model: {android_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
