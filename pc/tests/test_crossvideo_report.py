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
