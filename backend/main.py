from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.enterprise_router import enterprise_router
from backend.api.router import api_router
from backend.core.config import settings
from backend.db.seed import seed_initial_data
from backend.db.session import SessionLocal, init_db


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    db = SessionLocal()
    try:
        seed_initial_data(db)
    finally:
        db.close()
    yield


app = FastAPI(title=settings.PROJECT_NAME, version="2.5.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1")
app.include_router(enterprise_router, prefix="/api/v1")


@app.get("/")
def root():
    return {
        "message": "Welcome to the AI Packaging Advisor Enterprise API",
        "version": "2.5.0",
    }
