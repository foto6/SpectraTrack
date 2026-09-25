from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path

from .integrity import verify_sha256


@dataclass(slots=True)
class ModelManifest:
    name: str
    sha256: str
    input_size: int = 640
    source: str = ""
    format: str = "YOLO ONNX xywh+class-scores"
    notes: str = ""

    @classmethod
    def load(cls, path: str | Path) -> "ModelManifest":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8")

    def verify(self, model_path: str | Path) -> str:
        if self.input_size <= 0:
            raise ValueError("manifest input_size must be positive")
        return verify_sha256(model_path, self.sha256)
