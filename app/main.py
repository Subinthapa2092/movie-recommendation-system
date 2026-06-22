"""
Movie Recommendation System — FastAPI Backend
Serves content-based, collaborative (SVD), and hybrid recommendations
from pre-trained saved models.
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager

# Add project root to path so src.* imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.router import router
from app.state import load_models_into_state

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading saved models...")
    load_models_into_state(app)
    print("Models loaded. API ready.")
    yield

app = FastAPI(
    title="Movie Recommendation System",
    description="Hybrid recommender: TF-IDF content-based + SVD collaborative filtering",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")

# Serve frontend static files
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.isdir(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_frontend():
        return FileResponse(os.path.join(frontend_dir, "index.html"))

@app.get("/ping")
async def health_check():
    return {"status": "ok", "message": "Movie Recommendation API is running"}