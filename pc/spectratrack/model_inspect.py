from __future__ import annotations

import argparse
import json
from pathlib import Path

import onnxruntime as ort

from .integrity import sha256_file
from .model_manifest import ModelManifest


def _shape_text(shape) -> str:
    return "[" + ", ".join(str(x) for x in shape) + "]"


def main() -> int:
    p = argparse.ArgumentParser(description="Inspect a SpectraTrack ONNX model")
    p.add_argument("model")
    p.add_argument("--manifest-out", default="")
    p.add_argument("--name", default="")
    p.add_argument("--source", default="")
    p.add_argument("--input-size", type=int, default=0)
    args = p.parse_args()

    model = Path(args.model)
    if not model.exists():
        raise SystemExit(f"Model not found: {model}")

    digest = sha256_file(model)
    session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])

    print(f"path={model}")
    print(f"sha256={digest}")
    print(f"available_providers={','.join(ort.get_available_providers())}")
    for i, inp in enumerate(session.get_inputs()):
        print(f"input[{i}] name={inp.name} type={inp.type} shape={_shape_text(inp.shape)}")
    for i, out in enumerate(session.get_outputs()):
        print(f"output[{i}] name={out.name} type={out.type} shape={_shape_text(out.shape)}")

    if args.manifest_out:
        detected_size = args.input_size
        if detected_size <= 0:
            shape = session.get_inputs()[0].shape
            if len(shape) == 4 and isinstance(shape[2], int) and shape[2] == shape[3]:
                detected_size = int(shape[2])
        if detected_size <= 0:
            raise SystemExit("Cannot infer fixed square input size; pass --input-size")
        manifest = ModelManifest(
            name=args.name or model.stem,
            sha256=digest,
            input_size=detected_size,
            source=args.source or "inspected local ONNX file",
        )
        manifest.save(args.manifest_out)
        print(f"manifest={args.manifest_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
