from spectratrack.tracking_eval import evaluate


def test_synthetic_identity_eval_has_good_match_rate():
    result = evaluate(frames=180, targets=5, seed=7, use_appearance=True)
    assert result.match_rate > 0.75
    assert result.id_switches <= 3


def test_eval_is_deterministic():
    a = evaluate(frames=80, targets=4, seed=99, use_appearance=True)
    b = evaluate(frames=80, targets=4, seed=99, use_appearance=True)
    assert a == b
