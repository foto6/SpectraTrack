from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .model_security import normalize_sha256, sha256_file


@dataclass(slots=True)
class ModelManifest:
    name: str
    sha256: str
    source: str = ""
    license: str = ""
    input_size: int | None = None
    output_layout: str = ""
    schema_version: int = 1

    def __post_init__(self) -> None:
        self.name = str(self.name).strip()
        if not self.name:
            raise ValueError("Model manifest name is required")
        self.sha256 = normalize_sha256(self.sha256)
        if self.input_size is not None:
            self.input_size = int(self.input_size)
            if self.input_size <= 0:
                raise ValueError("input_size must be positive")

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "sha256": self.sha256,
            "source": self.source,
            "license": self.license,
            "input_size": self.input_size,
            "output_layout": self.output_layout,
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ModelManifest":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if int(data.get("schema_version", 1)) != 1:
            raise ValueError(f"Unsupported manifest schema: {data.get('schema_version')}")
        return cls(
            name=data["name"],
            sha256=data["sha256"],
            source=data.get("source", ""),
            license=data.get("license", ""),
            input_size=data.get("input_size"),
            output_layout=data.get("output_layout", ""),
            schema_version=1,
        )

    def verify(self, model_path: str | Path) -> str:
        actual = sha256_file(model_path)
        if actual != self.sha256:
            raise ValueError(
                f"Model manifest SHA-256 mismatch: expected {self.sha256}, got {actual}"
            )
        return actual
