import numpy as np
from collections import deque, defaultdict
from src.schemas import Track


class SpeedEstimator:
    """
    Computes speed from net displacement over a real-time window (not
    adjacent-frame differences), which is robust to fps changes and to
    per-frame bbox jitter. Also applies exponential smoothing so the
    displayed number doesn't jump frame-to-frame — same stability profile
    as DistanceEstimator, which reads a direct value with no differencing.
    """

    def __init__(self, focal_length_px: float, window_seconds: float = 0.5,
                 min_movement_m: float = 0.03, ema_alpha: float = 0.3):
        self.focal_length_px = focal_length_px
        self.window_seconds = window_seconds
        self.min_movement_m = min_movement_m   # ignore jitter below this — treat as stationary
        self.ema_alpha = ema_alpha             # smoothing factor for the final displayed speed
        self._history: dict[int, deque] = defaultdict(lambda: deque(maxlen=200))
        self._smoothed_speed: dict[int, float] = {}

    def update(self, track: Track) -> Track:
        if track.distance_m is None:
            return track

        x1, y1, x2, y2 = track.bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        now = track.timestamp

        history = self._history[track.track_id]
        history.append((now, cx, cy, track.distance_m))

        # drop samples older than the window — true rolling time window, independent of fps
        while history and (now - history[0][0]) > self.window_seconds:
            history.popleft()

        if len(history) < 2:
            track.speed_kmh = self._smoothed_speed.get(track.track_id, 0.0)
            return track

        t0, x0, y0, d0 = history[0]
        t1, x1_, y1_, d1 = history[-1]
        dt = t1 - t0

        if dt <= 0:
            track.speed_kmh = self._smoothed_speed.get(track.track_id, 0.0)
            return track

        dx_m = ((x1_ - x0) / self.focal_length_px) * d0
        dy_m = ((y1_ - y0) / self.focal_length_px) * d0
        dz_m = d1 - d0
        dist_m = float(np.sqrt(dx_m**2 + dy_m**2 + dz_m**2))

        raw_speed = 0.0 if dist_m < self.min_movement_m else (dist_m / dt) * 3.6

        prev = self._smoothed_speed.get(track.track_id, raw_speed)
        smoothed = self.ema_alpha * raw_speed + (1 - self.ema_alpha) * prev
        self._smoothed_speed[track.track_id] = smoothed

        track.speed_kmh = round(smoothed, 2)
        return track

    def reset_track(self, track_id: int) -> None:
        self._history.pop(track_id, None)
        self._smoothed_speed.pop(track_id, None)