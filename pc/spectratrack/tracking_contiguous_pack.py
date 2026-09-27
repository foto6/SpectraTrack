from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

PACK_SCHEMA = "spectratrack-tracking-contiguous-pack-v1"
PACK_REVISION = "mot17-04-contiguous-smoke400-r1"
REPLAY_SCHEMA = "spectratrack-detection-replay-v1"
OBSERVATION_SCHEMA = "spectratrack-detection-prefusion-v1"
EXPECTED_VIDEO = "golden/public/mot17/MOT17-04"
EXPECTED_WINDOWS = ((0, 199), (400, 599))
EXPECTED_SELECTED_FRAMES = 400
EXPECTED_SOURCE_FRAMES = 600
EXPECTED_CORPUS_REVISION = "mot17-public-r1"
EXPECTED_CORPUS_SHA256 = "8bfa6e54ab7a0160c133d8c7d0a2896b7ba254a23cf06afee2d4c9c453836759"
EXPECTED_SELECTION_MANIFEST_SHA256 = "cc26aa3a37f5830912e576d9475d83b231b1842519bcd9a22baab8c06c3c4844"
EXPECTED_SOURCE_REPLAY_SHA256 = "b2719a2c93c353123437497ac9513033898d3d65750a32ed633d05e8fb65b2d4"
EXPECTED_SOURCE_REPLAY_CANONICAL_SHA256 = "d27d45172a2df0a5c81ff83be2ceed03d2ed1eef922d3fde06f72b3126a89c59"
EXPECTED_SOURCE_GT_SHA256 = "28dcb9d197e0a098a1efb097f1589177350192a8f5f1be3e2ab5cd18d8f205c7"
EXPECTED_SOURCE_COMMIT = "943abee566c45116cee2b0e72b7d2c48753891ef"
EXPECTED_MODEL_SHA256 = "e84cbad768b218d74ecc85e3e52d84631123719a6951b3ddf6eddc850d5b3f73"
EXPECTED_PROVIDER = "DmlExecutionProvider,CPUExecutionProvider"


def _canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl_records(path: str | Path) -> Iterable[dict[str, Any]]:
    source = Path(path)
    with source.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw in enumerate(handle, start=1):
            stripped = raw.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{source}:{line_number}: invalid JSON") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{source}:{line_number}: expected JSON object")
            yield value


def _read_selection_manifest(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    value = json.loads(source.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError("tracking selection manifest must be a JSON object")
    if value.get("schema") != "spectratrack-tracking-smoke-manifest-v1":
        raise ValueError("tracking selection manifest schema mismatch")
    if value.get("video") != EXPECTED_VIDEO:
        raise ValueError("tracking selection manifest video mismatch")
    if value.get("source_frame_count") != EXPECTED_SOURCE_FRAMES:
        raise ValueError("tracking selection manifest must describe exactly 600 source frames")
    if value.get("selected_frame_count") != EXPECTED_SELECTED_FRAMES:
        raise ValueError("tracking selection manifest must select exactly 400 frames")
    if value.get("source_replay_file_sha256") != EXPECTED_SOURCE_REPLAY_SHA256:
        raise ValueError("tracking selection source replay SHA-256 changed")
    if value.get("source_replay_canonical_sha256") != EXPECTED_SOURCE_REPLAY_CANONICAL_SHA256:
        raise ValueError("tracking selection canonical replay SHA-256 changed")
    if value.get("ground_truth_sha256") != EXPECTED_SOURCE_GT_SHA256:
        raise ValueError("tracking selection ground-truth SHA-256 changed")
    selection = value.get("selection")
    if not isinstance(selection, dict):
        raise ValueError("tracking selection policy is missing")
    if selection.get("tracker_reset_between_windows") is not True:
        raise ValueError("tracking selection must reset tracker between windows")
    if selection.get("candidate_outcome_independent") is not True:
        raise ValueError("tracking selection must remain candidate-outcome independent")
    windows = value.get("windows")
    if not isinstance(windows, list) or len(windows) != len(EXPECTED_WINDOWS):
        raise ValueError("tracking selection manifest must contain exactly two windows")
    for ordinal, ((expected_start, expected_end), window) in enumerate(zip(EXPECTED_WINDOWS, windows)):
        if not isinstance(window, dict):
            raise ValueError("tracking selection window must be an object")
        expected_ids = list(range(expected_start, expected_end + 1))
        if window.get("ordinal") != ordinal:
            raise ValueError("tracking selection window ordinal mismatch")
        if window.get("start_position") != expected_start or window.get("end_position") != expected_end:
            raise ValueError("tracking selection window positions changed")
        if window.get("frame_ids") != expected_ids:
            raise ValueError("tracking selection frame IDs changed")
    return value


def _selection_frame_ids(selection: dict[str, Any]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(item) for item in window["frame_ids"]) for window in selection["windows"])


def _frame_ids_sha256(frame_ids: Iterable[int]) -> str:
    return _sha256_bytes(_canonical_json_bytes(list(frame_ids)))


def _canonicalize_records(records: Iterable[dict[str, Any]]) -> bytes:
    payload = bytearray()
    for record in records:
        payload.extend(_canonical_json_bytes(record))
    return bytes(payload)


def _split_selected_stream(
    path: str | Path,
    *,
    schema: str,
    video: str,
    windows: tuple[tuple[int, ...], ...],
    source_name: str,
    allow_trailing_records: bool = False,
) -> tuple[dict[str, Any], tuple[tuple[dict[str, Any], ...], ...]]:
    """Parse only metadata plus selected frame positions from the frozen 600-frame source."""
    source = Path(path)
    selected_by_position = {
        position: frame_id
        for frame_ids in windows
        for position, frame_id in zip(frame_ids, frame_ids)
    }
    by_frame: dict[int, dict[str, Any]] = {}
    metadata: dict[str, Any] | None = None
    source_position = 0
    trailing_records = 0

    with source.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw in enumerate(handle, start=1):
            stripped = raw.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if metadata is None:
                try:
                    value = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{source}:{line_number}: invalid metadata JSON") from exc
                if not isinstance(value, dict) or value.get("type") != "metadata":
                    raise ValueError(f"{source}:{line_number}: first record must be metadata")
                if value.get("schema") != schema:
                    raise ValueError(f"{source_name} schema mismatch")
                if value.get("video") != video:
                    raise ValueError(f"{source_name} metadata video mismatch")
                metadata = value
                continue

            if source_position >= EXPECTED_SOURCE_FRAMES:
                trailing_records += 1
                if not allow_trailing_records:
                    raise ValueError(
                        f"{source_name} contains records beyond the frozen "
                        f"{EXPECTED_SOURCE_FRAMES}-frame source"
                    )
                continue

            expected_frame = selected_by_position.get(source_position)
            if expected_frame is not None:
                try:
                    value = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{source}:{line_number}: invalid selected-frame JSON"
                    ) from exc
                if not isinstance(value, dict) or value.get("type") != "frame":
                    raise ValueError(
                        f"{source}:{line_number}: selected {source_name} record must be frame"
                    )
                frame = value.get("frame")
                if frame != expected_frame:
                    raise ValueError(
                        f"{source_name} selected position {source_position} "
                        f"has frame {frame!r}, expected {expected_frame}"
                    )
                if value.get("video") != video:
                    raise ValueError(f"selected {source_name} frame video mismatch")
                if frame in by_frame:
                    raise ValueError(f"duplicate selected {source_name} frame {frame}")
                by_frame[frame] = value
            # Excluded source positions are deliberately not JSON-decoded.
            source_position += 1

    if metadata is None:
        raise ValueError(f"{source_name} metadata is missing")
    if source_position != EXPECTED_SOURCE_FRAMES:
        raise ValueError(
            f"{source_name} has {source_position} source frame records; "
            f"expected {EXPECTED_SOURCE_FRAMES}"
        )
    if allow_trailing_records and trailing_records > 1:
        raise ValueError(f"{source_name} has unexpected trailing records")

    output: list[tuple[dict[str, Any], ...]] = []
    for window in windows:
        missing = [frame for frame in window if frame not in by_frame]
        if missing:
            raise ValueError(f"{source_name} is missing selected frames: {missing[:10]}")
        output.append(tuple(by_frame[frame] for frame in window))
    return metadata, tuple(output)


def _split_replay(
    path: str | Path,
    *,
    video: str,
    windows: tuple[tuple[int, ...], ...],
) -> tuple[dict[str, Any], tuple[tuple[dict[str, Any], ...], ...]]:
    return _split_selected_stream(
        path,
        schema=REPLAY_SCHEMA,
        video=video,
        windows=windows,
        source_name="replay",
    )


def _split_ground_truth(
    path: str | Path,
    *,
    video: str,
    windows: tuple[tuple[int, ...], ...],
) -> tuple[tuple[dict[str, Any], ...], ...]:
    wanted = {frame_id for window in windows for frame_id in window}
    by_frame: dict[int, dict[str, Any]] = {}
    for record in _jsonl_records(path):
        if record.get("video") != video:
            continue
        frame = record.get("frame")
        if isinstance(frame, bool) or not isinstance(frame, int):
            raise ValueError("ground-truth frame must be an integer")
        if frame in wanted:
            if frame in by_frame:
                raise ValueError(f"duplicate selected ground-truth frame {frame}")
            by_frame[frame] = record
    output = []
    for window in windows:
        missing = [frame for frame in window if frame not in by_frame]
        if missing:
            raise ValueError(f"ground truth is missing selected frames: {missing[:10]}")
        output.append(tuple(by_frame[frame] for frame in window))
    return tuple(output)


def _split_observations(
    path: str | Path,
    *,
    video: str,
    windows: tuple[tuple[int, ...], ...],
) -> tuple[dict[str, Any], tuple[tuple[dict[str, Any], ...], ...]]:
    return _split_selected_stream(
        path,
        schema=OBSERVATION_SCHEMA,
        video=video,
        windows=windows,
        source_name="observations",
        allow_trailing_records=True,
    )


def _payload_entry(path: Path, frame_count: int, record_count: int) -> dict[str, Any]:
    return {
        "path": path.name,
        "sha256": _sha256_path(path),
        "bytes": path.stat().st_size,
        "frame_count": frame_count,
        "record_count": record_count,
    }


def _manifest_digest(payload: dict[str, Any]) -> str:
    clone = dict(payload)
    clone.pop("pack_digest", None)
    return _sha256_bytes(_canonical_json_bytes(clone))


def build_contiguous_pack(
    *,
    selection_manifest_path: str | Path,
    replay_path: str | Path,
    ground_truth_path: str | Path,
    observations_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    if _sha256_path(selection_manifest_path) != EXPECTED_SELECTION_MANIFEST_SHA256:
        raise ValueError("selection manifest SHA-256 does not match frozen smoke gate")
    selection = _read_selection_manifest(selection_manifest_path)
    if _sha256_path(replay_path) != selection.get("source_replay_file_sha256"):
        raise ValueError("source replay file SHA-256 does not match frozen selection")
    if _sha256_path(ground_truth_path) != selection.get("ground_truth_sha256"):
        raise ValueError("source ground-truth SHA-256 does not match frozen selection")

    windows = _selection_frame_ids(selection)
    replay_metadata, replay_windows = _split_replay(replay_path, video=EXPECTED_VIDEO, windows=windows)
    gt_windows = _split_ground_truth(ground_truth_path, video=EXPECTED_VIDEO, windows=windows)
    observation_metadata, observation_windows = _split_observations(
        observations_path,
        video=EXPECTED_VIDEO,
        windows=windows,
    )

    for field in ("video", "model_sha256", "provider", "width", "height"):
        if replay_metadata.get(field) != observation_metadata.get(field):
            raise ValueError(f"replay/observation metadata mismatch: {field}")
    if replay_metadata.get("source_commit") != EXPECTED_SOURCE_COMMIT:
        raise ValueError("replay source commit mismatch")
    if observation_metadata.get("source_commit") != EXPECTED_SOURCE_COMMIT:
        raise ValueError("observation source commit mismatch")
    if replay_metadata.get("model_sha256") != EXPECTED_MODEL_SHA256:
        raise ValueError("replay model SHA-256 mismatch")
    if replay_metadata.get("provider") != EXPECTED_PROVIDER:
        raise ValueError("replay provider mismatch")

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    window_entries: list[dict[str, Any]] = []
    all_frame_ids: list[int] = []

    for ordinal, ((start, end), frame_ids, replay_rows, gt_rows, observation_rows) in enumerate(
        zip(EXPECTED_WINDOWS, windows, replay_windows, gt_windows, observation_windows)
    ):
        stem = f"mot17-04-{start:03d}-{end:03d}"
        replay_out = target / f"{stem}.replay.jsonl"
        gt_out = target / f"{stem}.gt.jsonl"
        observation_out = target / f"{stem}.observations.jsonl"

        replay_out.write_bytes(_canonicalize_records((replay_metadata, *replay_rows)))
        gt_out.write_bytes(_canonicalize_records(gt_rows))
        observation_out.write_bytes(_canonicalize_records((observation_metadata, *observation_rows)))

        all_frame_ids.extend(frame_ids)
        window_entries.append(
            {
                "ordinal": ordinal,
                "start_frame": start,
                "end_frame": end,
                "frame_count": len(frame_ids),
                "frame_ids_sha256": _frame_ids_sha256(frame_ids),
                "payloads": {
                    "replay": _payload_entry(replay_out, len(frame_ids), len(frame_ids) + 1),
                    "ground_truth": _payload_entry(gt_out, len(frame_ids), len(frame_ids)),
                    "observations": _payload_entry(observation_out, len(frame_ids), len(frame_ids) + 1),
                },
            }
        )

    manifest = {
        "schema": PACK_SCHEMA,
        "revision": PACK_REVISION,
        "research_only": True,
        "video": EXPECTED_VIDEO,
        "control_tracker": "current MultiObjectTracker",
        "tracker_policy_changed": False,
        "source": {
            "selection_manifest_path": Path(selection_manifest_path).name,
            "selection_manifest_sha256": _sha256_path(selection_manifest_path),
            "source_replay_file_sha256": selection["source_replay_file_sha256"],
            "source_replay_canonical_sha256": selection["source_replay_canonical_sha256"],
            "source_ground_truth_sha256": selection["ground_truth_sha256"],
            "source_observations_sha256": _sha256_path(observations_path),
            "source_observation_schema": OBSERVATION_SCHEMA,
            "source_replay_schema": REPLAY_SCHEMA,
            "corpus_revision": EXPECTED_CORPUS_REVISION,
            "corpus_sha256": EXPECTED_CORPUS_SHA256,
            "replay_source_commit": replay_metadata.get("source_commit"),
            "observation_source_commit": observation_metadata.get("source_commit"),
            "model_sha256": replay_metadata.get("model_sha256"),
            "provider": replay_metadata.get("provider"),
        },
        "selection": {
            "source_frame_count": EXPECTED_SOURCE_FRAMES,
            "selected_frame_count": len(all_frame_ids),
            "window_count": len(window_entries),
            "frame_ids_sha256": _frame_ids_sha256(all_frame_ids),
            "windows": [[start, end] for start, end in EXPECTED_WINDOWS],
            "excluded_middle_window": [200, 399],
            "tracker_reset_between_windows": True,
            "nightowls_tracking_gt_used": False,
        },
        "windows": window_entries,
    }
    manifest["pack_digest"] = _manifest_digest(manifest)
    manifest_path = target / "manifest.json"
    manifest_path.write_bytes((json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    return manifest


def _frame_ids_from_payload(path: Path, *, expected_kind: str) -> tuple[int, ...]:
    frame_ids: list[int] = []
    metadata_count = 0
    for record in _jsonl_records(path):
        kind = record.get("type")
        if expected_kind in {"replay", "observations"}:
            if kind == "metadata":
                metadata_count += 1
                continue
            if kind != "frame":
                raise ValueError(f"{path}: unexpected {expected_kind} record type {kind!r}")
        frame = record.get("frame")
        if isinstance(frame, bool) or not isinstance(frame, int):
            raise ValueError(f"{path}: payload frame must be an integer")
        frame_ids.append(frame)
    if expected_kind in {"replay", "observations"} and metadata_count != 1:
        raise ValueError(f"{path}: expected exactly one metadata record")
    return tuple(frame_ids)


def validate_contiguous_pack(pack_dir: str | Path) -> dict[str, Any]:
    root = Path(pack_dir)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != PACK_SCHEMA:
        raise ValueError("contiguous pack schema mismatch")
    if manifest.get("revision") != PACK_REVISION:
        raise ValueError("contiguous pack revision mismatch")
    if manifest.get("video") != EXPECTED_VIDEO:
        raise ValueError("contiguous pack video mismatch")
    if manifest.get("control_tracker") != "current MultiObjectTracker":
        raise ValueError("contiguous pack control tracker changed")
    if manifest.get("tracker_policy_changed") is not False:
        raise ValueError("contiguous pack must not change tracker policy")
    if manifest.get("pack_digest") != _manifest_digest(manifest):
        raise ValueError("contiguous pack digest mismatch")

    selection = manifest.get("selection")
    if not isinstance(selection, dict):
        raise ValueError("contiguous pack selection missing")
    if selection.get("source_frame_count") != EXPECTED_SOURCE_FRAMES:
        raise ValueError("contiguous pack source frame count mismatch")
    if selection.get("selected_frame_count") != EXPECTED_SELECTED_FRAMES:
        raise ValueError("contiguous pack selected frame count mismatch")
    if selection.get("windows") != [[start, end] for start, end in EXPECTED_WINDOWS]:
        raise ValueError("contiguous pack windows changed")
    if selection.get("excluded_middle_window") != [200, 399]:
        raise ValueError("contiguous pack leakage guard changed")
    if selection.get("nightowls_tracking_gt_used") is not False:
        raise ValueError("NightOwls must not be used as tracking GT")

    source = manifest.get("source")
    if not isinstance(source, dict):
        raise ValueError("contiguous pack source provenance missing")
    expected_source = {
        "selection_manifest_sha256": EXPECTED_SELECTION_MANIFEST_SHA256,
        "source_replay_file_sha256": EXPECTED_SOURCE_REPLAY_SHA256,
        "source_replay_canonical_sha256": EXPECTED_SOURCE_REPLAY_CANONICAL_SHA256,
        "source_ground_truth_sha256": EXPECTED_SOURCE_GT_SHA256,
        "corpus_revision": EXPECTED_CORPUS_REVISION,
        "corpus_sha256": EXPECTED_CORPUS_SHA256,
        "replay_source_commit": EXPECTED_SOURCE_COMMIT,
        "observation_source_commit": EXPECTED_SOURCE_COMMIT,
        "model_sha256": EXPECTED_MODEL_SHA256,
        "provider": EXPECTED_PROVIDER,
    }
    for key, expected in expected_source.items():
        if source.get(key) != expected:
            raise ValueError(f"contiguous pack frozen source provenance changed: {key}")

    windows = manifest.get("windows")
    if not isinstance(windows, list) or len(windows) != 2:
        raise ValueError("contiguous pack must contain exactly two windows")
    all_ids: list[int] = []
    for ordinal, ((start, end), window) in enumerate(zip(EXPECTED_WINDOWS, windows)):
        expected_ids = tuple(range(start, end + 1))
        if window.get("ordinal") != ordinal:
            raise ValueError("contiguous pack window ordinal mismatch")
        if window.get("start_frame") != start or window.get("end_frame") != end:
            raise ValueError("contiguous pack window boundary mismatch")
        if window.get("frame_count") != len(expected_ids):
            raise ValueError("contiguous pack window frame count mismatch")
        if window.get("frame_ids_sha256") != _frame_ids_sha256(expected_ids):
            raise ValueError("contiguous pack window frame-ID hash mismatch")

        payloads = window.get("payloads")
        if not isinstance(payloads, dict):
            raise ValueError("contiguous pack payload table missing")
        observed_sets: list[tuple[int, ...]] = []
        for kind in ("replay", "ground_truth", "observations"):
            entry = payloads.get(kind)
            if not isinstance(entry, dict):
                raise ValueError(f"contiguous pack payload missing: {kind}")
            payload_path = root / str(entry.get("path"))
            if not payload_path.is_file():
                raise ValueError(f"contiguous pack payload file missing: {payload_path.name}")
            if _sha256_path(payload_path) != entry.get("sha256"):
                raise ValueError(f"contiguous pack payload SHA mismatch: {payload_path.name}")
            if payload_path.stat().st_size != entry.get("bytes"):
                raise ValueError(f"contiguous pack payload byte count mismatch: {payload_path.name}")
            frame_ids = _frame_ids_from_payload(payload_path, expected_kind=kind)
            if frame_ids != expected_ids:
                raise ValueError(f"contiguous pack {kind} frame leakage or ordering mismatch")
            if entry.get("frame_count") != len(frame_ids):
                raise ValueError(f"contiguous pack {kind} frame count metadata mismatch")
            expected_records = len(frame_ids) + (1 if kind in {"replay", "observations"} else 0)
            if entry.get("record_count") != expected_records:
                raise ValueError(f"contiguous pack {kind} record count metadata mismatch")
            observed_sets.append(frame_ids)
        if not (observed_sets[0] == observed_sets[1] == observed_sets[2]):
            raise ValueError("contiguous pack replay/GT/observation frame sets differ")
        all_ids.extend(expected_ids)

    if any(200 <= frame <= 399 for frame in all_ids):
        raise ValueError("contiguous pack leaked frames from 200-399")
    if selection.get("frame_ids_sha256") != _frame_ids_sha256(all_ids):
        raise ValueError("contiguous pack aggregate frame-ID hash mismatch")
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build/validate the A2 canonical contiguous MOT17 replay pack")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--selection-manifest", required=True)
    build.add_argument("--replay", required=True)
    build.add_argument("--ground-truth", required=True)
    build.add_argument("--observations", required=True)
    build.add_argument("--output-dir", required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("--pack-dir", required=True)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if args.command == "build":
        manifest = build_contiguous_pack(
            selection_manifest_path=args.selection_manifest,
            replay_path=args.replay,
            ground_truth_path=args.ground_truth,
            observations_path=args.observations,
            output_dir=args.output_dir,
        )
        print(json.dumps({
            "pack_digest": manifest["pack_digest"],
            "selected_frame_count": manifest["selection"]["selected_frame_count"],
            "source_observations_sha256": manifest["source"]["source_observations_sha256"],
        }, indent=2, sort_keys=True))
        return 0
    manifest = validate_contiguous_pack(args.pack_dir)
    print(json.dumps({
        "pack_digest": manifest["pack_digest"],
        "selected_frame_count": manifest["selection"]["selected_frame_count"],
        "windows": manifest["selection"]["windows"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
