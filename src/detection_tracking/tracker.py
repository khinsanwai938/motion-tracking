import os
import time
from ultralytics import YOLO
import numpy as np
from src.schemas import Track

PERSON_CLASS_ID = 0  # COCO "person"


class PersonTracker:
    """
    Wraps YOLO26 + Ultralytics' built-in tracker (ByteTrack by default).
    Handles detection + persistent ID assignment in a single call.
    """

    def __init__(self, model_path: str = "yolo26n.pt", tracker_cfg: str = "configs/bytetrack_custom.yaml",
                 conf_threshold: float = 0.25, max_people: int | None = None,
                 imgsz: int = 416,
                 primary_grace_frames: int = 15):
        self.model = YOLO(model_path)
        # fall back to Ultralytics' default tracker if our tuned config is missing
        self.tracker_cfg = tracker_cfg if os.path.exists(tracker_cfg) else "bytetrack.yaml"
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz  # 640 = more accurate/slower, 416 = good CPU compromise, 320 = fastest
        self.primary_grace_frames = primary_grace_frames
        self._primary_id: int | None = None
        self._missing = 0
        self.max_people = max_people  # None = track everyone detected; set 1 for single-person demo mode

    def update(self, frame: np.ndarray) -> list[Track]:
        results = self.model.track(
            frame,
            imgsz=self.imgsz,
            classes=[PERSON_CLASS_ID],
            conf=self.conf_threshold,
            tracker=self.tracker_cfg,
            persist=True,
            verbose=False,
        )[0]

        if results.boxes.id is None:
            return self._select([])

        tracks = []
        now = time.time()
        for box, track_id in zip(results.boxes, results.boxes.id):
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            tracks.append(Track(track_id=int(track_id), bbox=(x1, y1, x2, y2), timestamp=now))

        return self._select(tracks)

    @staticmethod
    def _area(t: Track) -> float:
        x1, y1, x2, y2 = t.bbox
        return (x2 - x1) * (y2 - y1)

    def _select(self, tracks: list[Track]) -> list[Track]:
        if not self.max_people:
            return tracks
        if self.max_people > 1:
            return sorted(tracks, key=self._area, reverse=True)[: self.max_people]

        # Single-person mode: stick with the same ID while it exists. If it vanishes, wait a few
        # frames (it may just be a blur/occlusion) before switching to the largest person in view.
        # (Old rule "lowest ID wins" could lock onto a ghost detection.)
        if not tracks:
            self._missing += 1
            return []
        current = next((t for t in tracks if t.track_id == self._primary_id), None)
        if current is not None:
            self._missing = 0
            return [current]
        self._missing += 1
        if self._primary_id is not None and self._missing <= self.primary_grace_frames:
            return []
        best = max(tracks, key=self._area)
        self._primary_id, self._missing = best.track_id, 0
        return [best]