from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    p = argparse.ArgumentParser(description='Export a YOLO model to a fixed 640x640 ONNX used by SpectraTrack')
    p.add_argument('--model', default='yolo11n.pt')
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--opset', type=int, default=17)
    args = p.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit('Install exporter first: pip install ultralytics onnx onnxslim') from exc

    root = Path(__file__).resolve().parents[1]
    model = YOLO(args.model)
    exported = Path(model.export(format='onnx', imgsz=args.imgsz, opset=args.opset, simplify=True, dynamic=False))
    out = root / 'models' / 'yolo11n.onnx'
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(exported, out)
    android_out = root / 'android' / 'app' / 'src' / 'main' / 'assets' / 'yolo11n.onnx'
    shutil.copy2(out, android_out)
    print(f'PC model:      {out}')
    print(f'Android model: {android_out}')
    print(f'SHA-256:       {sha256(out)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
