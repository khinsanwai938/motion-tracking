import asyncio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()


@router.websocket("/ws/tracks")
async def track_stream(websocket: WebSocket):
    await websocket.accept()
    worker = websocket.app.state.worker
    try:
        while True:
            result = worker.get_latest(timeout=1.0)
            if result:
                await websocket.send_json(result)
            await asyncio.sleep(0.05)  # ~20 updates/sec to the client
    except WebSocketDisconnect:
        pass