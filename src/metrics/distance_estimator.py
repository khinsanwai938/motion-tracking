import threading
from typing import TYPE_CHECKING
from src.schemas import Track

if TYPE_CHECKING:  # depth model pulls in torch/transformers - only import it when actually used
    from src.metrics.depth_model import MonocularDepthEstimator


class DistanceEstimator:
    """
    Reads distance from a depth map at each track's centroid.

    The depth model is the slowest part of the pipeline, so it runs in a
    background thread: set_frame() hands it a frame and returns immediately
    (if the model is still busy, that frame is simply skipped). The main loop
    keeps using the most recent finished depth map, so video never freezes
    while depth is being computed.
    """

    def __init__(self, depth_model: "MonocularDepthEstimator"):
        self.depth_model = depth_model
        self._current_depth_map = None
        self._busy = threading.Event()
        self.last_depth_ms = 0.0

    def set_frame(self, frame) -> None:
        if self._busy.is_set():
            return  # previous depth job still running - skip, don't queue
        self._busy.set()
        threading.Thread(target=self._compute, args=(frame.copy(),), daemon=True).start()

    def _compute(self, frame) -> None:
        import time
        t0 = time.time()
        try:
            self._current_depth_map = self.depth_model.estimate_depth_map(frame)
            self.last_depth_ms = (time.time() - t0) * 1000
        finally:
            self._busy.clear()

    def estimate(self, track: Track) -> Track:
        depth_map = self._current_depth_map
        if depth_map is None:
            return track

        x1, y1, x2, y2 = track.bbox
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        track.distance_m = round(self.depth_model.read_depth_at(depth_map, cx, cy), 2)
        return track



class HeightDistanceEstimator:
    """
    CPU-friendly distance: no neural network, costs ~0 ms.

    Pinhole camera geometry:  distance = focal_length_px * real_size_m / size_in_pixels
      * full body visible (box not cut by the top/bottom edge) -> use box HEIGHT, assume `person_height_m`
      * body cut by the frame edge (e.g. sitting at a laptop)  -> use box WIDTH,  assume `body_width_m`
    The result is smoothed per track so the number doesn't jitter.

    Accuracy is roughly +/-15-30 %: people differ in size, and arms/pose change the box.
    Tune person_height_m / body_width_m, or set a real focal length in configs/camera_webcam.yaml.
    Same interface as DistanceEstimator, so the pipeline doesn't care which one it gets.
    """

    def __init__(self, focal_length_px: float, person_height_m: float = 1.7,
                 body_width_m: float = 0.5, ema_alpha: float = 0.4, edge_margin_px: int = 3):
        self.focal_length_px = focal_length_px
        self.person_height_m = person_height_m
        self.body_width_m = body_width_m
        self.ema_alpha = ema_alpha
        self.edge_margin_px = edge_margin_px
        self._frame_h = None
        self._smoothed: dict[int, float] = {}
        self.last_depth_ms = 0.0  # kept so the FPS overlay works with either estimator

    def set_frame(self, frame) -> None:
        self._frame_h = frame.shape[0]

    def estimate(self, track: Track) -> Track:
        x1, y1, x2, y2 = track.bbox
        h_px, w_px = y2 - y1, x2 - x1
        if h_px < 2 or w_px < 2:
            return track

        cut = self._frame_h is not None and (
            y1 <= self.edge_margin_px or y2 >= self._frame_h - self.edge_margin_px)
        raw = (self.focal_length_px * self.body_width_m / w_px) if cut \
            else (self.focal_length_px * self.person_height_m / h_px)

        prev = self._smoothed.get(track.track_id, raw)
        d = self.ema_alpha * raw + (1 - self.ema_alpha) * prev
        self._smoothed[track.track_id] = d
        track.distance_m = round(d, 2)
        return track