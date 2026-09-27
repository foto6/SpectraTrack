from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from ..qa_benchmark import GroundTruthFrame


SELECTOR_SCHEMA = "spectratrack-frozen-frame-selector-v1"


@dataclass(frozen=True, slots=True)
class FrozenFrameSelector:
    path: str
    manifest_sha256: str
    selection_sha256: str
    revision: str
    source_corpus_revision: str | None
    source_corpus_sha256: str | None
    frame_ids: tuple[tuple[str, int], ...]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_frame_id(value: Any, where: str) -> tuple[str, int]:
    if isinstance(value, str):
        video, separator, frame_text = value.rpartition("#")
        if not separator or not video or not frame_text.isdigit():
            raise ValueError(f"{where}: string frame id must be '<video>#<non-negative-frame>'")
        frame = int(frame_text)
        return video, frame
    if not isinstance(value, dict):
        raise ValueError(f"{where}: frame id must be an object or '<video>#<frame>' string")
    video = value.get("video")
    frame = value.get("frame")
    if not isinstance(video, str) or not video:
        raise ValueError(f"{where}: video must be a non-empty string")
    if not isinstance(frame, int) or isinstance(frame, bool) or frame < 0:
        raise ValueError(f"{where}: frame must be a non-negative integer")
    return video, frame


def _selection_hash(frame_ids: Iterable[tuple[str, int]]) -> str:
    canonical = [
        {"frame": frame, "video": video}
        for video, frame in sorted(frame_ids)
    ]
    payload = json.dumps(canonical, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_frozen_frame_selector(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_count: int,
    expected_revision: str | None = None,
) -> FrozenFrameSelector:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"frame manifest not found: {source}")
    actual_sha256 = _sha256_file(source)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"frame manifest SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    if expected_count <= 0:
        raise ValueError("expected_count must be > 0")

    data = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("frame manifest must be a JSON object")
    revision = data.get("revision")
    if not isinstance(revision, str) or not revision:
        raise ValueError("frame manifest revision must be a non-empty string")
    if expected_revision is not None and revision != expected_revision:
        raise ValueError(
            f"frame manifest revision mismatch: expected {expected_revision!r}, got {revision!r}"
        )

    schema = data.get("schema")
    proof_schema = "spectratrack-smoke-selection-proof-v1"
    if schema not in {None, SELECTOR_SCHEMA, proof_schema}:
        raise ValueError(f"unsupported frame selector schema: {schema!r}")

    raw_frames = data.get("frame_ids", data.get("frames"))
    if schema == proof_schema:
        raw_frames = data.get("entries")
        if not isinstance(raw_frames, list):
            raise ValueError("smoke selection proof must contain an entries list")
        frame_ids = tuple(
            _parse_frame_id(
                {"video": item.get("video"), "frame": item.get("smoke_frame")}
                if isinstance(item, dict) else item,
                f"frame manifest[{index}]",
            )
            for index, item in enumerate(raw_frames)
        )
    else:
        if not isinstance(raw_frames, list):
            raise ValueError("frame manifest must contain a frame_ids or frames list")
        frame_ids = tuple(
            _parse_frame_id(item, f"frame manifest[{index}]")
            for index, item in enumerate(raw_frames)
        )
    if len(frame_ids) != expected_count:
        raise ValueError(
            f"frame manifest count mismatch: expected {expected_count}, got {len(frame_ids)}"
        )
    if len(set(frame_ids)) != len(frame_ids):
        raise ValueError("frame manifest contains duplicate frame ids")

    parent = data.get("parent") if isinstance(data.get("parent"), dict) else {}
    source_revision = data.get("source_corpus_revision", parent.get("revision"))
    if source_revision is not None and (not isinstance(source_revision, str) or not source_revision):
        raise ValueError("source_corpus_revision must be a non-empty string when present")
    source_sha = data.get("source_corpus_sha256", parent.get("corpus_sha256"))
    if source_sha is not None and (
        not isinstance(source_sha, str)
        or len(source_sha) != 64
        or any(char not in "0123456789abcdefABCDEF" for char in source_sha)
    ):
        raise ValueError("source_corpus_sha256 must be a 64-character hex string when present")

    return FrozenFrameSelector(
        path=str(source),
        manifest_sha256=actual_sha256,
        selection_sha256=(
            str(data.get("selection", {}).get("selection_identity_sha256"))
            if schema == proof_schema
            and isinstance(data.get("selection"), dict)
            and isinstance(data["selection"].get("selection_identity_sha256"), str)
            else _selection_hash(frame_ids)
        ),
        revision=revision,
        source_corpus_revision=source_revision,
        source_corpus_sha256=source_sha.lower() if isinstance(source_sha, str) else None,
        frame_ids=frame_ids,
    )


def select_ground_truth_frames(
    annotations: Iterable[GroundTruthFrame],
    selector: FrozenFrameSelector,
) -> list[GroundTruthFrame]:
    by_id = {(item.video, item.frame): item for item in annotations}
    selected_ids = set(selector.frame_ids)
    missing = selected_ids - set(by_id)
    if missing:
        preview = [f"{video}#{frame}" for video, frame in sorted(missing)[:12]]
        raise ValueError(f"frame manifest references ids absent from ground truth: {preview}")
    selected = [by_id[key] for key in selector.frame_ids]
    if len(selected) != len(selector.frame_ids):
        raise ValueError("frame manifest selection is not one-to-one")
    return selected


def selector_record(selector: FrozenFrameSelector) -> dict[str, Any]:
    return {
        "path": selector.path,
        "revision": selector.revision,
        "manifest_sha256": selector.manifest_sha256,
        "selection_sha256": selector.selection_sha256,
        "frame_count": len(selector.frame_ids),
        "source_corpus_revision": selector.source_corpus_revision,
        "source_corpus_sha256": selector.source_corpus_sha256,
    }
