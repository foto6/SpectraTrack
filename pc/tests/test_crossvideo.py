from spectratrack.crossvideo import (
    TrackletSummary,
    build_cross_video_graph,
    descriptor_similarity,
    same_object_score,
)


def t(video, tid, label, cls, descriptor, **kwargs):
    return TrackletSummary(
        video=video,
        local_track_id=tid,
        class_id=cls,
        label=label,
        first_frame=10,
        last_frame=40,
        observations=8,
        mean_score=0.9,
        mean_quality=0.85,
        best_frame=24,
        fps=30.0,
        descriptor=descriptor,
        **kwargs,
    )


def test_descriptor_similarity_identity():
    assert descriptor_similarity((1.0, 0.0), (1.0, 0.0)) == 1.0


def test_strong_cross_video_match_forms_entity():
    graph = build_cross_video_graph([
        t("a.mp4", 1, "car", 2, (1.0, 0.0, 0.0)),
        t("b.mp4", 4, "car", 2, (0.999, 0.02, 0.0)),
    ])
    assert len(graph["entities"]) == 1
    assert len(graph["entities"][0]["members"]) == 2
    assert graph["edges"][0]["strength"] == "strong"
    assert graph["edges"][0]["relation"] == "same_object_candidate"
    assert graph["entities"][0]["global_object_id"] == graph["entities"][0]["entity_id"]
    assert {m["local_track_id"] for m in graph["entities"][0]["members"]} == {1, 4}
    assert len({m["global_object_id"] for m in graph["entities"][0]["members"]}) == 1


def test_different_classes_never_link():
    graph = build_cross_video_graph([
        t("a.mp4", 1, "car", 2, (1.0, 0.0)),
        t("b.mp4", 2, "bird", 14, (1.0, 0.0)),
    ])
    assert graph["edges"] == []
    assert len(graph["entities"]) == 2


def test_two_same_class_tracks_from_same_video_do_not_merge():
    graph = build_cross_video_graph([
        t("a.mp4", 1, "car", 2, (1.0, 0.0)),
        t("a.mp4", 2, "car", 2, (1.0, 0.0)),
    ])
    assert len(graph["entities"]) == 2


def test_person_semantics_are_appearance_not_identity():
    graph = build_cross_video_graph([
        t("a.mp4", 1, "person", 0, (1.0, 0.0)),
        t("b.mp4", 2, "person", 0, (1.0, 0.0)),
    ])
    assert graph["edges"][0]["relation"] == "same_appearance_candidate"
    assert graph["entities"][0]["semantics"] == "appearance_group_not_identity"
    assert graph["entities"][0]["entity_id"].startswith("PERSON_APPEARANCE_")


def test_review_edge_does_not_force_entity_merge():
    graph = build_cross_video_graph(
        [
            t("a.mp4", 1, "car", 2, (1.0, 0.0)),
            t("b.mp4", 2, "car", 2, (0.90, 0.435889894)),
        ],
        candidate_threshold=0.85,
        strong_threshold=0.95,
    )
    assert len(graph["edges"]) == 1
    assert graph["edges"][0]["strength"] == "review"
    assert len(graph["entities"]) == 2


def test_complete_link_prevents_similarity_chain_overmerge():
    graph = build_cross_video_graph(
        [
            t("a.mp4", 1, "car", 2, (1.0, 0.0)),
            t("b.mp4", 2, "car", 2, (0.98, 0.199)),
            t("c.mp4", 3, "car", 2, (0.88, 0.475)),
        ],
        candidate_threshold=0.80,
        strong_threshold=0.93,
    )
    sizes = sorted(len(entity["members"]) for entity in graph["entities"])
    assert sizes == [1, 2]


def test_multi_signal_score_is_not_appearance_only():
    left = t(
        "a.mp4",
        1,
        "car",
        2,
        (1.0, 0.0),
        color_descriptor=(1.0, 0.0),
        mean_aspect_ratio=1.0,
        mean_relative_area=0.01,
    )
    right = t(
        "b.mp4",
        2,
        "car",
        2,
        (1.0, 0.0),
        color_descriptor=(0.0, 1.0),
        mean_aspect_ratio=4.0,
        mean_relative_area=0.20,
    )
    score, signals = same_object_score(left, right)
    assert signals["appearance"] == 1.0
    assert signals["color"] == 0.0
    assert score is not None
    assert score < 0.70


def test_non_overlapping_same_video_fragments_can_share_global_object():
    common = {
        "color_descriptor": (1.0, 0.0),
        "mean_aspect_ratio": 1.8,
        "mean_relative_area": 0.04,
        "motion_direction": (1.0, 0.0),
    }
    first = TrackletSummary(
        video="a.mp4",
        local_track_id=3,
        class_id=2,
        label="car",
        first_frame=1,
        last_frame=10,
        observations=8,
        mean_score=0.9,
        mean_quality=0.85,
        best_frame=5,
        fps=30.0,
        descriptor=(1.0, 0.0),
        start_center=(0.20, 0.50),
        end_center=(0.35, 0.50),
        **common,
    )
    second = TrackletSummary(
        video="a.mp4",
        local_track_id=9,
        class_id=2,
        label="car",
        first_frame=14,
        last_frame=24,
        observations=9,
        mean_score=0.9,
        mean_quality=0.85,
        best_frame=18,
        fps=30.0,
        descriptor=(1.0, 0.0),
        start_center=(0.40, 0.50),
        end_center=(0.55, 0.50),
        **common,
    )
    graph = build_cross_video_graph([first, second])
    assert len(graph["entities"]) == 1
    members = graph["entities"][0]["members"]
    assert {member["local_track_id"] for member in members} == {3, 9}
    assert len({member["global_object_id"] for member in members}) == 1
    assert graph["edges"][0]["signals"]["temporal"] is not None
    assert graph["edges"][0]["signals"]["direction"] is not None
    assert graph["scoring"]["probability_calibrated"] is False


def test_overlapping_same_video_tracks_remain_incompatible():
    left = t("a.mp4", 1, "car", 2, (1.0, 0.0))
    right = t("a.mp4", 2, "car", 2, (1.0, 0.0))
    graph = build_cross_video_graph([left, right])
    assert len(graph["entities"]) == 2
    assert graph["edges"] == []
