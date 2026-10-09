import time
from src.schemas import Track
from src.metrics.speed_estimator import SpeedEstimator


def test_stationary_person_has_zero_speed():
    estimator = SpeedEstimator(focal_length_px=1000)
    t = Track(track_id=1, bbox=(100, 100, 200, 300), distance_m=3.0)
    for _ in range(5):
        t.timestamp = time.time()
        t = estimator.update(t)
        time.sleep(0.05)
    assert t.speed_kmh < 1.0  # should stay near zero, not drift