from fastapi import APIRouter, Request
from api.models import HealthResponse, TrackingSnapshot

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health(request: Request):
    worker = request.app.state.worker
    return HealthResponse(status="ok", worker_running=worker._thread.is_alive())


@router.get("/tracks/latest", response_model=TrackingSnapshot | None)
def latest_tracks(request: Request):
    worker = request.app.state.worker
    return worker.get_latest(timeout=0.5)