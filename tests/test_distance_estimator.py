from src.schemas import Track
from src.metrics.distance_estimator import DistanceEstimator


def test_distance_decreases_as_bbox_grows():
    estimator = DistanceEstimator(focal_length_px=1000)
    near = estimator.estimate(Track(track_id=1, bbox=(100, 50, 300, 450)))  # tall bbox = close
    far = estimator.estimate(Track(track_id=1, bbox=(100, 50, 200, 150)))   # short bbox = far
    assert near.distance_m < far.distance_m


def test_zero_height_bbox_does_not_crash():
    estimator = DistanceEstimator(focal_length_px=1000)
    t = estimator.estimate(Track(track_id=1, bbox=(100, 100, 200, 100)))
    assert t.distance_m is None