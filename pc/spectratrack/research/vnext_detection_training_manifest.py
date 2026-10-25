from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

SCHEMA = "spectratrack-detection-training-manifest-v1"
REQUIRED_FORBIDDEN_EVAL_SETS = {
    "nightowls-public-r1",
    "nightowls-smoke400",
    "nightowls-holdout4200",
    "nightowls-full5000",
}
FORBIDDEN_SOURCE_TOKENS = {
    "nightowls",
    "smoke400",
    "holdout4200",
    "heldout4200",
    "full5000",
    "slice5000",
}
ALLOWED_ROLES = {"positive", "negative", "hard_negative"}
ALLOWED_SELECTION_ORIGINS = {"human", "teacher_candidate", "baseline_fp"}
POSITIVE_CONDITION_TAGS = {"night", "small", "occluded"}


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(ch in "0123456789abcdef" for ch in value.lower())
    )


def _validate_bbox(value: Any, where: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"{where}: bbox must be [x1,y1,x2,y2]")
    try:
        box = [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{where}: bbox values must be numeric") from exc
    if not all(math.isfinite(item) for item in box):
        raise ValueError(f"{where}: bbox values must be finite")
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError(f"{where}: bbox must have positive area")
    return box


def _source_text(item: Mapping[str, Any]) -> str:
    parts = [
        item.get("source_dataset"),
        item.get("source_revision"),
        item.get("source_path"),
        item.get("video"),
        item.get("item_id"),
    ]
    return " ".join(str(value).lower() for value in parts if value is not None)


def _canonical_sample_key(item: Mapping[str, Any]) -> str:
    bbox = item.get("bbox")
    bbox_key = ""
    if isinstance(bbox, list):
        bbox_key = ",".join(f"{float(value):.6f}" for value in bbox)
    raw = "|".join(
        [
            str(item.get("source_sha256", "")),
            str(item.get("video", "")),
            str(item.get("frame", "")),
            bbox_key,
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def validate_manifest_data(data: Mapping[str, Any]) -> dict[str, Any]:
    if data.get("schema") != SCHEMA:
        raise ValueError(f"manifest schema must be {SCHEMA!r}")
    if data.get("purpose") != "detector_finetune":
        raise ValueError("manifest purpose must be detector_finetune")
    revision = data.get("revision")
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError("manifest revision must be a non-empty string")
    created_from_commit = data.get("created_from_commit")
    if not isinstance(created_from_commit, str) or len(created_from_commit) != 40:
        raise ValueError("created_from_commit must be an exact 40-character git SHA")

    forbidden = data.get("forbidden_evaluation_sets")
    if not isinstance(forbidden, list) or not all(isinstance(item, str) and item for item in forbidden):
        raise ValueError("forbidden_evaluation_sets must be a list of non-empty strings")
    missing = REQUIRED_FORBIDDEN_EVAL_SETS - set(forbidden)
    if missing:
        raise ValueError(
            "manifest must prelock all required eval exclusions: " + ", ".join(sorted(missing))
        )

    items = data.get("items")
    if not isinstance(items, list):
        raise ValueError("manifest items must be a list")

    item_ids: set[str] = set()
    sample_keys: set[str] = set()
    counts = {role: 0 for role in sorted(ALLOWED_ROLES)}
    tag_counts: dict[str, int] = {}
    for index, item in enumerate(items):
        where = f"items[{index}]"
        if not isinstance(item, Mapping):
            raise ValueError(f"{where}: item must be an object")

        item_id = item.get("item_id")
        if not isinstance(item_id, str) or not item_id:
            raise ValueError(f"{where}: item_id must be non-empty")
        if item_id in item_ids:
            raise ValueError(f"{where}: duplicate item_id {item_id!r}")
        item_ids.add(item_id)

        role = item.get("role")
        if role not in ALLOWED_ROLES:
            raise ValueError(f"{where}: role must be one of {sorted(ALLOWED_ROLES)}")
        counts[str(role)] += 1

        selection_origin = item.get("selection_origin")
        if selection_origin not in ALLOWED_SELECTION_ORIGINS:
            raise ValueError(
                f"{where}: selection_origin must be one of {sorted(ALLOWED_SELECTION_ORIGINS)}"
            )

        for field in ("source_dataset", "source_revision", "source_path", "video"):
            value = item.get(field)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{where}: {field} must be non-empty")

        source_text = _source_text(item)
        matched_tokens = sorted(token for token in FORBIDDEN_SOURCE_TOKENS if token in source_text)
        if matched_tokens:
            raise ValueError(
                f"{where}: evaluation leakage forbidden; matched tokens {matched_tokens}"
            )

        frame = item.get("frame")
        if not isinstance(frame, int) or isinstance(frame, bool) or frame < 0:
            raise ValueError(f"{where}: frame must be a non-negative integer")

        if not _is_sha256(item.get("source_sha256")):
            raise ValueError(f"{where}: source_sha256 must be SHA-256")
        if not _is_sha256(item.get("image_sha256")):
            raise ValueError(f"{where}: image_sha256 must be SHA-256")

        tags = item.get("tags")
        if not isinstance(tags, list) or not all(isinstance(tag, str) and tag for tag in tags):
            raise ValueError(f"{where}: tags must be a list of non-empty strings")
        tags_set = set(tags)
        for tag in tags_set:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

        label = item.get("label")
        bbox = item.get("bbox")
        if role == "positive":
            if label != "person":
                raise ValueError(f"{where}: positive item label must be person")
            _validate_bbox(bbox, where)
            if not (tags_set & POSITIVE_CONDITION_TAGS):
                raise ValueError(
                    f"{where}: positive item must include at least one of "
                    f"{sorted(POSITIVE_CONDITION_TAGS)}"
                )
        else:
            if label != "background":
                raise ValueError(f"{where}: {role} item label must be background")
            if bbox is not None:
                _validate_bbox(bbox, where)
            if "negative" not in tags_set:
                raise ValueError(f"{where}: {role} item must include negative tag")

        key = _canonical_sample_key(item)
        if key in sample_keys:
            raise ValueError(
                f"{where}: duplicate source/frame/bbox sample across manifest roles"
            )
        sample_keys.add(key)

    return {
        "schema": SCHEMA,
        "revision": revision,
        "item_count": len(items),
        "counts_by_role": counts,
        "tag_counts": dict(sorted(tag_counts.items())),
        "forbidden_evaluation_sets": sorted(forbidden),
        "status": "VALID_DISJOINT_TRAINING_MANIFEST",
    }


def validate_manifest_file(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    data = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError("manifest root must be an object")
    return validate_manifest_data(data)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate SpectraTrack vNext detector fine-tune/hard-negative manifests"
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output")
    return parser


def main() -> int:
    args = _parser().parse_args()
    summary = validate_manifest_file(args.manifest)
    text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
