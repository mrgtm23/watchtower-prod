from fastapi import FastAPI
from backend.app.core.config import settings
from backend.app.db.session import Base, engine
from backend.app.api.v1 import uploads, auth, predict, monitoring, training
from prometheus_client import make_asgi_app
from fastapi.middleware.cors import CORSMiddleware
import os

# Create DB tables if missing (for dev; use migrations in prod)
Base.metadata.create_all(bind=engine)

app = FastAPI(title=settings.PROJECT_NAME)

app.include_router(auth.router, prefix="/api/v1/users", tags=["users"])
app.include_router(uploads.router, prefix="/api/v1/models", tags=["models"])
app.include_router(predict.router, prefix="/api/v1", tags=["models"])
app.include_router(monitoring.router, prefix="/api/v1/monitoring", tags=["monitoring"])
app.include_router(training.router, prefix="/api/v1/training", tags=["training"])

# mount prometheus metrics at /metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)