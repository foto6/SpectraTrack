from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .detection_replay import DetectionReplay, ReplayFrame, load_detection_replay
from .qa_benchmark import GroundTruthFrame, ground_truth_sha256, load_ground_truth
from .tracking_research import (
    _aggregate,
    ambiguity_guard_promotion_gate,
    run_scored_replay_candidate,
    scenario_from_canonical_ground_truth,
)

TRACKING_SMOKE_MANIFEST_SCHEMA = "spectratrack-tracking-smoke-manifest-v1"
TRACKING_SMOKE_RESULT_SCHEMA = "spectratrack-tracking-smoke-result-v1"
DEFAULT_TARGET_FRAMES = 400
DEFAULT_WINDOW_FRAMES = 200


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_path(path: str | Path) -> str:
    return _sha256_bytes(Path(path).read_bytes())


def result_sha256(payload: dict[str, Any]) -> str:
    return _sha256_bytes(_json_bytes(payload))


def _window_starts(frame_count: int, target_frames: int, window_frames: int) -> tuple[int, ...]:
    if target_frames < 1 or window_frames < 1:
        raise ValueError("target_frames and window_frames must be positive")
    if frame_count < 1:
        raise ValueError("tracking smoke requires at least one replay frame")
    if frame_count <= target_frames:
        return (0,)
    if target_frames % window_frames:
        raise ValueError("target_frames must be divisible by window_frames")
    window_count = target_frames // window_frames
    if window_count < 1:
        raise ValueError("target_frames must include at least one window")
    if frame_count < window_count * window_frames:
        raise ValueError("replay is too short for non-overlapping smoke windows")
    if window_count == 1:
        return ((frame_count - window_frames) // 2,)
    span = frame_count - window_frames
    starts = tuple(round(index * span / (window_count - 1)) for index in range(window_count))
    if any(right - left < window_frames for left, right in zip(starts, starts[1:])):
        raise ValueError("deterministic smoke windows would overlap")
    return starts


def build_tracking_smoke_manifest(
    replay: DetectionReplay,
    ground_truth_hash: str,
    *,
    target_frames: int = DEFAULT_TARGET_FRAMES,
    window_frames: int = DEFAULT_WINDOW_FRAMES,
) -> dict[str, Any]:
    frames = replay.frames
    starts = _window_starts(len(frames), target_frames, window_frames)
    if len(frames) <= target_frames:
        windows = ((0, len(frames)),)
    else:
        windows = tuple((start, window_frames) for start in starts)
    selected = sum(length for _, length in windows)
    return {
        "schema": TRACKING_SMOKE_MANIFEST_SCHEMA,
        "video": replay.metadata.video,
        "source_replay_file_sha256": replay.source_sha256,
        "source_replay_canonical_sha256": replay.canonical_sha256(),
        "ground_truth_sha256": ground_truth_hash,
        "source_frame_count": len(frames),
        "selected_frame_count": selected,
        "target_frame_count": target_frames,
        "window_frame_count": window_frames,
        "selection": {
            "method": "evenly-spaced-contiguous-windows-v1",
            "candidate_outcome_independent": True,
            "tracker_reset_between_windows": True,
            "full_replay_mutated": False,
        },
        "windows": [
            {
                "ordinal": ordinal,
                "start_position": start,
                "end_position": start + length - 1,
                "frame_ids": [frame.frame for frame in frames[start : start + length]],
            }
            for ordinal, (start, length) in enumerate(windows)
        ],
    }


def validate_tracking_smoke_manifest(
    replay: DetectionReplay,
    manifest: dict[str, Any],
    *,
    ground_truth_hash: str,
) -> None:
    if manifest.get("schema") != TRACKING_SMOKE_MANIFEST_SCHEMA:
        raise ValueError(
            f"unsupported tracking smoke manifest schema: {manifest.get('schema')!r}"
        )
    expected = {
        "video": replay.metadata.video,
        "source_replay_file_sha256": replay.source_sha256,
        "source_replay_canonical_sha256": replay.canonical_sha256(),
        "ground_truth_sha256": ground_truth_hash,
        "source_frame_count": len(replay.frames),
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"tracking smoke manifest {key} does not match canonical input")
    windows = manifest.get("windows")
    if not isinstance(windows, list) or not windows:
        raise ValueError("tracking smoke manifest windows must be a non-empty list")
    seen: set[int] = set()
    selected = 0
    for ordinal, window in enumerate(windows):
        if not isinstance(window, dict) or window.get("ordinal") != ordinal:
            raise ValueError("tracking smoke windows must be ordered and zero-based")
        start = window.get("start_position")
        end = window.get("end_position")
        frame_ids = window.get("frame_ids")
        if not isinstance(start, int) or not isinstance(end, int) or end < start:
            raise ValueError("tracking smoke window positions are invalid")
        if not isinstance(frame_ids, list) or not frame_ids:
            raise ValueError("tracking smoke window frame_ids must be non-empty")
        actual = [frame.frame for frame in replay.frames[start : end + 1]]
        if frame_ids != actual:
            raise ValueError(
                "tracking smoke window frame_ids do not match canonical replay order"
            )
        if any(frame_id in seen for frame_id in frame_ids):
            raise ValueError("tracking smoke windows must not overlap")
        seen.update(frame_ids)
        selected += len(frame_ids)
    if selected != manifest.get("selected_frame_count"):
        raise ValueError(
            "tracking smoke selected_frame_count does not match windows"
        )


def _replay_window(replay: DetectionReplay, frame_ids: list[int]) -> DetectionReplay:
    by_id = {frame.frame: frame for frame in replay.frames}
    frames: list[ReplayFrame] = []
    for frame_id in frame_ids:
        try:
            frames.append(by_id[frame_id])
        except KeyError as exc:
            raise ValueError(
                f"tracking smoke frame {frame_id} is absent from replay"
            ) from exc
    return DetectionReplay(
        metadata=replay.metadata,
        frames=tuple(frames),
        source_sha256=replay.source_sha256,
    )


def _build_window_scenarios(
    replay: DetectionReplay,
    ground_truth: tuple[GroundTruthFrame, ...],
    windows: list[dict[str, Any]],
) -> tuple[Any, ...]:
    scenarios = []
    for window in windows:
        subset = _replay_window(replay, window["frame_ids"])
        scenarios.append(
            scenario_from_canonical_ground_truth(subset, ground_truth)
        )
    return tuple(scenarios)


def _score_candidate_scenarios(
    scenarios: tuple[Any, ...],
    candidate: str,
) -> dict[str, float | int | None]:
    return _aggregate(
        run_scored_replay_candidate(scenario, candidate)["metrics"]
        for scenario in scenarios
    )


def tracking_smoke_gate(
    control: dict[str, float | int | None],
    candidate: dict[str, float | int | None],
) -> dict[str, Any]:
    promotion_signal = ambiguity_guard_promotion_gate(control, candidate)
    control_frag = int(control["fragmentations"])
    candidate_frag = int(candidate["fragmentations"])
    frag_increase = candidate_frag - control_frag
    frag_increase_fraction = (
        frag_increase / control_frag
        if control_frag > 0
        else (0.0 if candidate_frag == 0 else float("inf"))
    )
    recall_loss = max(
        0.0,
        float(control["tracking_recall"]) - float(candidate["tracking_recall"]),
    )
    gates = {
        "tracking_recall_loss_max_0_25pp": recall_loss <= 0.0025 + 1e-12,
        "no_id_switch_regression": (
            int(candidate["id_switches"]) <= int(control["id_switches"])
        ),
        "fragmentation_increase_max_5pct": frag_increase_fraction <= 0.05 + 1e-12,
        "no_false_track_increase": (
            int(candidate["false_track_creations"])
            <= int(control["false_track_creations"])
        ),
    }
    rejected = not all(gates.values())
    return {
        "classification": (
            "reject_obvious_harm" if rejected else "survivor_or_ambiguous"
        ),
        "decision": (
            "REJECT BEFORE FULL CANONICAL REPLAY"
            if rejected
            else "PROCEED TO FULL CANONICAL REPLAY"
        ),
        "gates": gates,
        "delta": promotion_signal["delta"],
        "full_promotion_gate_signal_only": {
            "passed": promotion_signal["passed"],
            "gates": promotion_signal["gates"],
        },
    }


def score_tracking_smoke(
    replay: DetectionReplay,
    ground_truth: Iterable[GroundTruthFrame],
    manifest: dict[str, Any],
    *,
    manifest_sha256: str,
    ground_truth_hash: str,
    subject_commit: str | None = None,
) -> dict[str, Any]:
    validate_tracking_smoke_manifest(
        replay,
        manifest,
        ground_truth_hash=ground_truth_hash,
    )
    ground_truth_rows = tuple(ground_truth)
    windows = manifest["windows"]
    scenarios = _build_window_scenarios(replay, ground_truth_rows, windows)
    control = _score_candidate_scenarios(scenarios, "current")
    candidate = _score_candidate_scenarios(
        scenarios,
        "current-ambiguity-guard",
    )
    selected_frames = int(manifest["selected_frame_count"])
    full_frames = int(manifest["source_frame_count"])
    candidate_count = 2
    smoke_steps = selected_frames * candidate_count
    full_steps = full_frames * candidate_count
    return {
        "schema": TRACKING_SMOKE_RESULT_SCHEMA,
        "subject_commit": subject_commit,
        "video": replay.metadata.video,
        "manifest_sha256": manifest_sha256,
        "source_replay_file_sha256": replay.source_sha256,
        "source_replay_canonical_sha256": replay.canonical_sha256(),
        "ground_truth_sha256": ground_truth_hash,
        "selected_frame_count": selected_frames,
        "source_frame_count": full_frames,
        "window_count": len(windows),
        "candidate_names": ["current", "current-ambiguity-guard"],
        "detector_policy_runs": 0,
        "onnx_inference_calls": 0,
        "control": control,
        "candidate": candidate,
        "smoke_gate": tracking_smoke_gate(control, candidate),
        "compute_savings_estimate": {
            "unit": "tracker_candidate_frame_steps",
            "smoke_steps": smoke_steps,
            "full_steps": full_steps,
            "saved_steps_if_rejected": full_steps - smoke_steps,
            "saved_fraction_if_rejected": 1.0 - (smoke_steps / full_steps),
        },
    }


def _write_json(path: str | Path, payload: dict[str, Any]) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(_json_bytes(payload))
    return _sha256_path(output)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SpectraTrack deterministic tracking smoke gate"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser(
        "freeze", help="Freeze a candidate-independent smoke manifest"
    )
    freeze.add_argument("--replay", required=True)
    freeze.add_argument("--ground-truth", required=True)
    freeze.add_argument(
        "--target-frames", type=int, default=DEFAULT_TARGET_FRAMES
    )
    freeze.add_argument(
        "--window-frames", type=int, default=DEFAULT_WINDOW_FRAMES
    )
    freeze.add_argument("--output", required=True)
    score = subparsers.add_parser(
        "score", help="Score current vs ambiguity guard on a frozen smoke"
    )
    score.add_argument("--replay", required=True)
    score.add_argument("--ground-truth", required=True)
    score.add_argument("--manifest", required=True)
    score.add_argument("--revision")
    score.add_argument("--output", required=True)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    replay = load_detection_replay(args.replay)
    gt_hash = ground_truth_sha256(args.ground_truth)
    if args.command == "freeze":
        manifest = build_tracking_smoke_manifest(
            replay,
            gt_hash,
            target_frames=args.target_frames,
            window_frames=args.window_frames,
        )
        artifact_hash = _write_json(args.output, manifest)
        print(f"manifest_sha256={artifact_hash}")
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    manifest_hash = _sha256_path(args.manifest)
    report = score_tracking_smoke(
        replay,
        load_ground_truth(args.ground_truth),
        manifest,
        manifest_sha256=manifest_hash,
        ground_truth_hash=gt_hash,
        subject_commit=args.revision,
    )
    artifact_hash = _write_json(args.output, report)
    print(f"result_sha256={artifact_hash}")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
