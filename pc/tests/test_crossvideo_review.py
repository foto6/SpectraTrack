from spectratrack.crossvideo import TrackletSummary, build_cross_video_graph


def _t(video, tid, descriptor=(1.0, 0.0)):
    return TrackletSummary(
        video=video,
        local_track_id=tid,
        class_id=2,
        label="car",
        first_frame=1,
        last_frame=10,
        observations=5,
        mean_score=0.9,
        mean_quality=0.8,
        best_frame=5,
        fps=30.0,
        descriptor=descriptor,
    )


def test_manual_different_blocks_auto_strong_merge():
    a = _t("a.mp4", 1)
    b = _t("b.mp4", 2)
    graph = build_cross_video_graph(
        [a, b],
        review_decisions={tuple(sorted((a.key, b.key))): "different"},
    )
    assert len(graph["entities"]) == 2
    assert graph["edges"][0]["review_decision"] == "different"
    assert graph["edges"][0]["strength"] == "manual"


def test_manual_same_can_merge_below_similarity_threshold():
    a = _t("a.mp4", 1, (1.0, 0.0))
    b = _t("b.mp4", 2, (0.2, 0.979795897))
    graph = build_cross_video_graph(
        [a, b],
        review_decisions={tuple(sorted((a.key, b.key))): "same"},
    )
    assert len(graph["entities"]) == 1
    assert len(graph["entities"][0]["members"]) == 2


def test_unsure_does_not_force_merge():
    a = _t("a.mp4", 1, (1.0, 0.0))
    b = _t("b.mp4", 2, (0.5, 0.8660254))
    graph = build_cross_video_graph(
        [a, b],
        candidate_threshold=0.4,
        strong_threshold=0.95,
        review_decisions={tuple(sorted((a.key, b.key))): "unsure"},
    )
    assert len(graph["entities"]) == 2
