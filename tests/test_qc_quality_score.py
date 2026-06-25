from forgery_pipeline.qc.quality_score import (
    qes_score, route_from_score, bucket_from_score, area_validity)


def test_weighted_sum():
    s = qes_score(1.0, 1.0, 1.0, 1.0, 1.0)
    assert abs(s - 1.0) < 1e-9
    s2 = qes_score(1.0, 0.0, 0.0, 0.0, 0.0)
    assert abs(s2 - 0.3) < 1e-9


def test_area_validity():
    assert area_validity(0.1) == 1.0
    assert area_validity(0.005) == 0.0
    assert area_validity(0.7) == 0.0


def test_routing_thresholds():
    assert route_from_score(0.80) == "accept"
    assert route_from_score(0.65) == "review"
    assert route_from_score(0.50) == "reject"
    assert bucket_from_score(0.80) == "high"
    assert bucket_from_score(0.65) == "mid"
    assert bucket_from_score(0.50) == "low"
