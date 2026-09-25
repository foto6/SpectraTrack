from __future__ import annotations

import argparse
import json
import hashlib
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    p = argparse.ArgumentParser(description="Create a SpectraTrack model provenance manifest")
    p.add_argument("model")
    p.add_argument("--name", default="")
    p.add_argument("--source", default="")
    p.add_argument("--license", default="")
    p.add_argument("--input-size", type=int, default=None)
    p.add_argument("--output-layout", default="xywh + class scores")
    p.add_argument("--output", default="")
    args = p.parse_args()

    model = Path(args.model)
    if not model.is_file():
        raise SystemExit(f"Model not found: {model}")
    output = Path(args.output) if args.output else model.with_suffix(model.suffix + ".manifest.json")
    payload = {
        "schema_version": 1,
        "name": args.name or model.stem,
        "sha256": sha256(model),
        "source": args.source,
        "license": args.license,
        "input_size": args.input_size,
        "output_layout": args.output_layout,
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)
    print(payload["sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
