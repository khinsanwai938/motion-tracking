from dataclasses import dataclass, field
from collections import deque
import time


@dataclass
class Track:
    track_id: int
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    timestamp: float = field(default_factory=time.time)
    distance_m: float | None = None
    speed_kmh: float | None = None
    name: str | None = None  # recognised person name (None = recognition off)
    history: deque = field(default_factory=lambda: deque(maxlen=30))