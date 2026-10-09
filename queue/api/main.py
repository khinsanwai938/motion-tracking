from contextlib import asynccontextmanager
from fastapi import FastAPI
from src.worker import PipelineWorker
from api import routes, ws


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.worker = PipelineWorker(single_person_mode=True)
    app.state.worker.start()
    yield
    app.state.worker.stop()


app = FastAPI(title="Human Tracking Service", lifespan=lifespan)
app.include_router(routes.router)
app.include_router(ws.router)