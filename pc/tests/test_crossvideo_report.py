from spectratrack.crossvideo import TrackletSummary, build_cross_video_graph
from spectratrack.crossvideo_report import write_html_report


def test_html_report_contains_review_controls_and_person_semantics(tmp_path):
    tracklets = [
        TrackletSummary(
            "a.mp4", 1, 0, "person", 1, 20, 5, 0.9, 0.8, 8, 30.0,
            (1.0, 0.0), "a_T001.jpg",
        ),
        TrackletSummary(
            "b.mp4", 2, 0, "person", 1, 20, 5, 0.9, 0.8, 8, 30.0,
            (1.0, 0.0), "b_T002.jpg",
        ),
    ]
    graph = build_cross_video_graph(tracklets)
    path = write_html_report(graph, tmp_path / "report.html")
    text = path.read_text(encoding="utf-8")
    assert "PERSON_APPEARANCE_" in text
    assert "LOOKS SAME" in text
    assert "not biometric identity" in text
    assert "localStorage" in text
    assert "function exportReview()" in text
    assert "spectratrack_review.json" in text
    assert "function safeStore(" in text
    assert "function safeLoad(" in text



def test_schema_v2_global_object_ids_and_manual_review_render(tmp_path):
    left = TrackletSummary(
        "cam/a.mp4", 1, 2, "car", 1, 20, 5, 0.9, 0.8, 8, 30.0,
        (1.0, 0.0), "a_T001.jpg",
        color_descriptor=(1.0, 0.0),
        mean_aspect_ratio=1.8,
        mean_relative_area=0.02,
        start_center=(0.2, 0.4),
        end_center=(0.3, 0.4),
        motion_direction=(1.0, 0.0),
    )
    right = TrackletSummary(
        "cam/b.mp4", 2, 2, "car", 1, 20, 5, 0.9, 0.8, 8, 30.0,
        (1.0, 0.0), "b_T002.jpg",
        color_descriptor=(1.0, 0.0),
        mean_aspect_ratio=1.8,
        mean_relative_area=0.02,
        start_center=(0.2, 0.4),
        end_center=(0.3, 0.4),
        motion_direction=(1.0, 0.0),
    )
    decision_key = tuple(sorted((left.key, right.key)))
    graph = build_cross_video_graph(
        [left, right],
        review_decisions={decision_key: "different"},
    )

    assert graph["schema_version"] == 2
    assert graph["review_decisions_applied"] == 1
    assert all("global_object_id" in tracklet for tracklet in graph["tracklets"])
    assert all("global_object_id" in entity for entity in graph["entities"])
    edge = next(edge for edge in graph["edges"] if edge["review_decision"] == "different")
    assert edge["same_object_score"] is not None
    assert "signals" in edge

    path = write_html_report(graph, tmp_path / "schema-v2-report.html")
    text = path.read_text(encoding="utf-8")
    assert "DIFFERENT" in text
    assert "same_object_candidate" in text
