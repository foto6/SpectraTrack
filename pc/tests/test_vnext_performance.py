import numpy as np
import pytest

from spectratrack.detector import YoloOnnxDetector, _tile_regions
from spectratrack.vnext_performance import (
    FrameSignals,
    SchedulerConfig,
    current_compute_bounds,
    global_discovery_bound_frames,
    runtime_rates,
    schedule_frame,
    simulate_scheduler,
)


class FakeSession:
    def __init__(self):
        self.calls = 0

    def run(self, _outputs, _feeds):
        self.calls += 1
        return [np.zeros((1, 84, 10), dtype=np.float32)]


def make_detector() -> YoloOnnxDetector:
    detector = YoloOnnxDetector.__new__(YoloOnnxDetector)
    detector.input_w = 640
    detector.input_h = 640
    detector.input_name = "images"
    detector.session = FakeSession()
    detector.labels = ["person"] + [f"c{i}" for i in range(1, 80)]
    detector.conf_threshold = 0.35
    detector.iou_threshold = 0.45
    detector.class_thresholds = {}
    detector.last_stage_ms = {}
    detector.last_inference_calls = 0
    return detector


@pytest.mark.parametrize(
    ("width", "height", "tiles", "minimum_calls", "maximum_calls"),
    [
        (1280, 720, 6, 7, 13),
        (1920, 1080, 8, 9, 17),
        (2560, 1440, 15, 16, 31),
        (3840, 2160, 32, 33, 65),
    ],
)
def test_current_compute_bounds_use_exact_production_geometry(
    width,
    height,
    tiles,
    minimum_calls,
    maximum_calls,
):
    bounds = current_compute_bounds(width, height)
    assert len(bounds["minimum"].tile_regions) == tiles
    assert bounds["minimum"].total_onnx_calls == minimum_calls
    assert bounds["maximum"].total_onnx_calls == maximum_calls


def test_1080p_exact_tile_regions_match_production_logic():
    assert _tile_regions(1920, 1080, 640, 0.20) == [
        (0, 0, 640, 640),
        (512, 0, 1152, 640),
        (1024, 0, 1664, 640),
        (1280, 0, 1920, 640),
        (0, 440, 640, 1080),
        (512, 440, 1152, 1080),
        (1024, 440, 1664, 1080),
        (1280, 440, 1920, 1080),
    ]


@pytest.mark.parametrize(
    ("width", "height", "expected_calls"),
    [
        (1280, 720, 7),
        (1920, 1080, 9),
        (2560, 1440, 16),
        (3840, 2160, 33),
    ],
)
def test_actual_people_recall_off_counters_match_theoretical_minimum(width, height, expected_calls):
    detector = make_detector()
    frame = np.full((height, width, 3), 180, dtype=np.uint8)

    detector.detect_people_recall(
        frame,
        person_threshold=0.12,
        tile_size=640,
        tile_overlap=0.20,
        enhancement_mode="off",
    )

    assert detector.last_inference_calls == expected_calls
    assert detector.session.calls == expected_calls


def test_actual_1080p_adaptive_counter_hits_minimum_on_clean_uniform_frame():
    detector = make_detector()
    frame = np.full((1080, 1920, 3), 180, dtype=np.uint8)

    detector.detect_people_recall(
        frame,
        person_threshold=0.12,
        tile_size=640,
        tile_overlap=0.20,
        enhancement_mode="adaptive",
    )

    assert detector.last_inference_calls == 9


def test_actual_1080p_adaptive_counter_hits_maximum_when_every_tile_is_enhanced():
    detector = make_detector()
    frame = np.full((1080, 1920, 3), 12, dtype=np.uint8)

    detector.detect_people_recall(
        frame,
        person_threshold=0.12,
        tile_size=640,
        tile_overlap=0.20,
        enhancement_mode="adaptive",
    )

    assert detector.last_inference_calls == 17


def test_runtime_rates_convert_wall_time_to_source_time_cost():
    rates = runtime_rates(frames=300, elapsed_s=1200.0, source_fps=30.0, onnx_calls=5100)
    assert rates["processing_fps"] == pytest.approx(0.25)
    assert rates["processing_seconds_per_source_second"] == pytest.approx(120.0)
    assert rates["onnx_calls_per_frame"] == pytest.approx(17.0)
    assert rates["onnx_calls_per_source_second"] == pytest.approx(510.0)


def test_budgeted_scheduler_enforces_hard_call_limits():
    config = SchedulerConfig(global_period=10, max_calls_per_frame=3, max_enhanced_calls_per_frame=1)
    signals = FrameSignals(track_rois=8, suspect_rois=8, enhancement_eligible_rois=8)

    enhanced_seen = False
    for frame_index in range(30):
        decision = schedule_frame(
            "BUDGETED_ADAPTIVE",
            frame_index,
            signals,
            tile_count=8,
            config=config,
        )
        assert decision.total_calls <= 3
        assert decision.enhanced_roi_calls <= 1
        enhanced_seen = enhanced_seen or decision.enhanced_roi_calls > 0

    assert enhanced_seen


def test_scene_change_and_camera_motion_force_global_rediscovery():
    config = SchedulerConfig(global_period=100, max_calls_per_frame=4)

    scene = schedule_frame(
        "TRACK_GUIDED",
        7,
        FrameSignals(scene_change=True),
        tile_count=8,
        config=config,
    )
    motion = schedule_frame(
        "TRACK_GUIDED",
        8,
        FrameSignals(camera_motion_trigger=True),
        tile_count=8,
        config=config,
    )

    assert scene.full_frame_calls == 1
    assert scene.global_rescan_reason == "scene_change"
    assert motion.full_frame_calls == 1
    assert motion.global_rescan_reason == "camera_motion"



def test_trigger_overrides_detector_cadence_for_immediate_global_rescan():
    config = SchedulerConfig(detector_every=5, global_period=100, max_calls_per_frame=3)
    decision = schedule_frame(
        "BUDGETED_ADAPTIVE",
        3,
        FrameSignals(scene_change=True, suspect_rois=1, enhancement_eligible_rois=1),
        tile_count=8,
        config=config,
    )

    assert decision.full_frame_calls == 1
    assert decision.global_rescan_reason == "scene_change"
    assert decision.reused_temporal_state is False

def test_periodic_global_discovery_is_bounded():
    config = SchedulerConfig(detector_every=2, global_period=15)
    assert global_discovery_bound_frames("COARSE_TO_FINE", config) == 2
    assert global_discovery_bound_frames("TRACK_GUIDED", config) == 30
    assert global_discovery_bound_frames("BUDGETED_ADAPTIVE", config) == 30


def test_scheduler_simulation_reports_compute_frontier_inputs():
    signals = FrameSignals(track_rois=2, suspect_rois=2, enhancement_eligible_rois=1)
    result = simulate_scheduler(
        "BUDGETED_ADAPTIVE",
        width=1920,
        height=1080,
        frames=300,
        source_fps=30.0,
        signals=signals,
        config=SchedulerConfig(global_period=10, max_calls_per_frame=3, max_enhanced_calls_per_frame=1),
    )

    assert result["tile_count"] == 8
    assert result["max_onnx_calls_on_any_frame"] <= 3
    assert result["periodic_global_discovery_bound_frames"] == 10
    assert result["periodic_global_discovery_bound_seconds"] == pytest.approx(1 / 3)
