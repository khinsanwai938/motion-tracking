from pydantic import BaseModel


class TrackResult(BaseModel):
    track_id: int
    bbox: tuple[float, float, float, float]
    distance_m: float | None = None
    speed_kmh: float | None = None
    name: str | None = None


class TrackingSnapshot(BaseModel):
    timestamp: float
    tracks: list[TrackResult]


class HealthResponse(BaseModel):
    status: str
    worker_running: bool