from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from .model_manifest import ModelManifest
from .model_security import sha256_file, verify_sha256


def probe_camera(index: int) -> dict:
    backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    try:
        if not cap.isOpened():
            return {"index": index, "opened": False}
        ok, frame = cap.read()
        return {
            "index": index,
            "opened": True,
            "read_ok": bool(ok),
            "width": int(frame.shape[1]) if ok else int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(frame.shape[0]) if ok else int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "reported_fps": float(cap.get(cv2.CAP_PROP_FPS)),
            "backend": cap.getBackendName() if hasattr(cap, "getBackendName") else "unknown",
        }
    finally:
        cap.release()


def inspect_model(path: str | Path, expected_sha256: str = "") -> dict:
    path = Path(path)
    digest = verify_sha256(path, expected_sha256) if expected_sha256 else sha256_file(path)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return {
        "path": str(path),
        "sha256": digest,
        "inputs": [{"name": x.name, "shape": x.shape, "type": x.type} for x in session.get_inputs()],
        "outputs": [{"name": x.name, "shape": x.shape, "type": x.type} for x in session.get_outputs()],
    }


def collect(model: str = "", expected_sha256: str = "", cameras: list[int] | None = None, manifest_path: str = "") -> dict:
    result = {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "onnxruntime": ort.__version__,
        "providers_available": ort.get_available_providers(),
    }
    if model:
        manifest = ModelManifest.load(manifest_path) if manifest_path else None
        if manifest is not None:
            manifest.verify(model)
            result["model_manifest"] = manifest.to_dict()
            expected_sha256 = manifest.sha256
        result["model"] = inspect_model(model, expected_sha256)
    if cameras:
        result["cameras"] = [probe_camera(i) for i in cameras]
    return result


def main() -> int:
    p = argparse.ArgumentParser(description="SpectraTrack PC environment/model/camera diagnostics")
    p.add_argument("--model", default="")
    p.add_argument("--model-sha256", default="")
    p.add_argument("--model-manifest", default="")
    p.add_argument("--camera", type=int, action="append", default=[])
    p.add_argument("--output", default="")
    args = p.parse_args()
    try:
        result = collect(args.model, args.model_sha256, args.camera, args.model_manifest)
    except Exception as exc:
        result = {"error": str(exc)}
        payload = json.dumps(result, indent=2, sort_keys=True)
        print(payload)
        return 2
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    if args.output:
        Path(args.output).write_text(payload + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
