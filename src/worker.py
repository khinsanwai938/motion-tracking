import threading
import queue
import time
from src.pipeline import build_pipeline


class PipelineWorker:
    """
    Runs the tracking pipeline in a background thread so it never blocks
    the web server's event loop. Publishes each frame's results onto a
    bounded queue; the API layer just reads from it.
    """

    def __init__(self, single_person_mode: bool = True, max_queue_size: int = 2):
        self._pipeline = build_pipeline(single_person_mode=single_person_mode)
        self._result_queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        self._pipeline.camera.release()

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            ok, frame = self._pipeline.camera.read()
            if not ok:
                time.sleep(0.05)
                continue

            tracks = self._pipeline.tracker.update(frame)
            tracks = [self._pipeline.distance_estimator.estimate(t) for t in tracks]
            tracks = [self._pipeline.speed_estimator.update(t) for t in tracks]
            if self._pipeline.identifier is not None:
                tracks = self._pipeline.identifier.identify(frame, tracks)

            payload = {
                "timestamp": time.time(),
                "tracks": [
                    {
                        "track_id": t.track_id,
                        "bbox": t.bbox,
                        "distance_m": t.distance_m,
                        "speed_kmh": t.speed_kmh,
                        "name": t.name,
                    }
                    for t in tracks
                ],
            }

            # drop the oldest frame if the consumer is slower than the pipeline —
            # we always want the latest state, not a backlog
            if self._result_queue.full():
                self._result_queue.get_nowait()
            self._result_queue.put(payload)

    def get_latest(self, timeout: float = 1.0) -> dict | None:
        try:
            return self._result_queue.get(timeout=timeout)
        except queue.Empty:
            return None