from fastapi import FastAPI
from backend.api.router import api_router
from backend.core.config import settings

app = FastAPI(title=settings.PROJECT_NAME, version="1.0.0")

app.include_router(api_router, prefix="/api/v1")

@app.get("/")
def root():
    return {"message": "Welcome to the AI Packaging Advisor API"}
