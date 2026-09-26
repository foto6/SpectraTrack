from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from spectratrack.integrity import sha256_file

PREFUSION_SCHEMA = "spectratrack-detection-prefusion-v1"
ROI_SCHEMA = "spectratrack-vnext-enhancement-roi-v1"
WEAK_MIN = 0.12
WEAK_MAX = 0.35


def _load_source_map(
    path: str | Path | None,
) -> tuple[dict[tuple[str, int], dict[str, Any]], str | None]:
    if path is None:
        return {}, None
    source_map: dict[tuple[str, int], dict[str, Any]] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue
            data = json.loads(raw)
            video = data.get("video")
            frame = data.get("frame")
            source = data.get("source")
            source_frame = data.get("source_frame")
            if not isinstance(video, str) or not video:
                raise ValueError(f"{path}:{line_number}: video must be non-empty")
            if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0:
                raise ValueError(f"{path}:{line_number}: frame must be a non-negative integer")
            if source is None:
                continue
            if not isinstance(source, str) or not source:
                raise ValueError(f"{path}:{line_number}: source must be a non-empty string")
            if source_frame is not None and (
                isinstance(source_frame, bool)
                or not isinstance(source_frame, int)
                or source_frame < 0
            ):
                raise ValueError(
                    f"{path}:{line_number}: source_frame must be a non-negative integer"
                )
            key = (video, frame)
            if key in source_map:
                raise ValueError(f"{path}:{line_number}: duplicate source mapping for {key}")
            source_map[key] = {"source": source, "source_frame": source_frame}
    return source_map, sha256_file(path)


def _candidate_sort_key(candidate: dict[str, Any]) -> tuple[Any, ...]:
    bbox = candidate.get("bbox", [])
    return (
        -float(candidate["score"]),
        str(candidate["source_id"]),
        tuple(float(value) for value in bbox),
    )


def _is_weak_tile_person(candidate: dict[str, Any]) -> bool:
    return (
        candidate.get("label", "").lower() == "person"
        and candidate.get("source_kind") == "tile"
        and WEAK_MIN <= float(candidate.get("score", -1.0)) < WEAK_MAX
        and isinstance(candidate.get("source_id"), str)
        and bool(candidate.get("source_id"))
        and isinstance(candidate.get("source_region"), list)
        and len(candidate["source_region"]) == 4
    )


def _support_json(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "bbox": [float(value) for value in candidate["bbox"]],
        "score": float(candidate["score"]),
        "class_id": int(candidate["class_id"]),
        "label": str(candidate["label"]),
    }


def build_manifest(
    prefusion_path: str | Path,
    output_path: str | Path,
    *,
    source_map_path: str | Path | None = None,
    frame_limit: int | None = None,
) -> dict[str, Any]:
    if frame_limit is not None and frame_limit <= 0:
        raise ValueError("frame_limit must be > 0 when supplied")

    source_map, source_map_sha = _load_source_map(source_map_path)
    metadata: dict[str, Any] | None = None
    frame_rows: list[dict[str, Any]] = []

    with Path(prefusion_path).open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            data = json.loads(raw)
            kind = data.get("type")
            if kind == "metadata":
                if metadata is not None:
                    raise ValueError("prefusion dump has multiple metadata records")
                if data.get("schema") != PREFUSION_SCHEMA:
                    raise ValueError(f"unsupported prefusion schema: {data.get('schema')!r}")
                metadata = dict(data)
                continue
            if kind != "frame":
                continue

            frame_index = int(data["frame"])
            if frame_limit is not None and frame_index >= frame_limit:
                continue
            video = str(data["video"])
            candidates = data.get("candidates", [])
            if not isinstance(candidates, list):
                raise ValueError(f"{prefusion_path}:{line_number}: candidates must be a list")

            weak = [candidate for candidate in candidates if _is_weak_tile_person(candidate)]
            if not weak:
                continue
            weak.sort(key=_candidate_sort_key)
            trigger = weak[0]
            source_id = str(trigger["source_id"])
            source_kind = str(trigger["source_kind"])
            source_region = [int(value) for value in trigger["source_region"]]
            same_source_weak = [
                candidate
                for candidate in weak
                if candidate.get("source_id") == source_id
                and candidate.get("source_kind") == source_kind
                and [int(value) for value in candidate.get("source_region", [])] == source_region
            ]
            same_source_weak.sort(key=_candidate_sort_key)

            key = (video, frame_index)
            row = {
                "type": "roi",
                "video": video,
                "frame": frame_index,
                "roi_id": f"{video}#{frame_index}#{source_id}",
                "bbox": source_region,
                "signals": {"weak_person": True},
                "source_kind": source_kind,
                "source_id": source_id,
                "raw_support": [_support_json(candidate) for candidate in same_source_weak],
                "trigger": _support_json(trigger),
            }
            if key in source_map:
                row["source"] = source_map[key]["source"]
                if source_map[key]["source_frame"] is not None:
                    row["source_frame"] = source_map[key]["source_frame"]
            elif source_map:
                raise ValueError(f"source map is missing selected frame {key}")
            frame_rows.append(row)

    if metadata is None:
        raise ValueError("prefusion dump is missing metadata")
    if not frame_rows:
        raise ValueError("prefusion dump contains no strict weak-person candidate frames")

    frame_rows.sort(key=lambda row: (row["video"], row["frame"]))
    selected_keys = [(row["video"], row["frame"]) for row in frame_rows]
    if len(selected_keys) != len(set(selected_keys)):
        raise ValueError("strict manifest selected more than one ROI for a source frame")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_metadata = {
        "type": "metadata",
        "schema": ROI_SCHEMA,
        "source": "A1 prefusion weak tile evidence; highest weak person score per source frame; no GT objects used",
        "selection": {
            "label": "person",
            "source_kind": "tile",
            "weak_score_min_inclusive": WEAK_MIN,
            "weak_score_max_exclusive": WEAK_MAX,
            "max_selected_rois_per_source_frame": 1,
            "order": "score_desc_then_source_id_then_bbox",
            "ground_truth_used_for_selection": False,
        },
        "prefusion": {
            "path": str(prefusion_path),
            "sha256": sha256_file(prefusion_path),
            "source_commit": metadata.get("source_commit"),
            "model_sha256": metadata.get("model_sha256"),
            "video_sha256": metadata.get("video_sha256"),
            "config": metadata.get("config", {}),
        },
        "source_map": {
            "path": str(source_map_path) if source_map_path is not None else None,
            "sha256": source_map_sha,
            "purpose": "frame bytes lookup only; object annotations are not used for ROI selection",
        },
        "selected_frames": len(frame_rows),
    }
    with output.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(manifest_metadata, sort_keys=True) + "\n")
        for row in frame_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    return {
        "selected_frames": len(frame_rows),
        "prefusion_sha256": manifest_metadata["prefusion"]["sha256"],
        "source_map_sha256": source_map_sha,
        "output_sha256": sha256_file(output),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build strict A3 weak-person ROI manifest from frozen A1 prefusion evidence"
    )
    parser.add_argument("--prefusion", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--source-map-ground-truth",
        help="Optional canonical A5 JSONL used only to copy frame source paths",
    )
    parser.add_argument("--frame-limit", type=int)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    result = build_manifest(
        args.prefusion,
        args.output,
        source_map_path=args.source_map_ground_truth,
        frame_limit=args.frame_limit,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
