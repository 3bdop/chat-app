from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from src.controls.api import router

app = FastAPI(
    title="Ebla ChatBot Assistant", openapi_url=None, docs_url=None, redoc_url=None
)
app.include_router(router)

BASE_DIR = Path(__file__).parent

# Mount static files (CSS, images)
app.mount("/static", StaticFiles(directory=BASE_DIR / "view/static"), name="static")

# Allow CORS for frontend applications
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
