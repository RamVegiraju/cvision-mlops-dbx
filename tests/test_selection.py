import pytest

from cv_mlops.evaluation import Candidate, pick_winner


def c(version, acc, p95, backbone="m", digest="d"):
    return Candidate(version=version, backbone=backbone, test_accuracy=acc, latency_p95_ms=p95, test_digest=digest)


def test_highest_accuracy_wins_when_gap_exceeds_tolerance():
    winner, reason = pick_winner([c("1", 0.90, 10), c("2", 0.95, 50)], accuracy_tolerance=0.01)
    assert winner.version == "2"
    assert "highest test accuracy" in reason


def test_faster_model_wins_within_tolerance():
    winner, reason = pick_winner([c("1", 0.948, 10), c("2", 0.950, 50)], accuracy_tolerance=0.005)
    assert winner.version == "1"
    assert "fastest" in reason


def test_pick_winner_requires_candidates():
    with pytest.raises(ValueError):
        pick_winner([], 0.01)

