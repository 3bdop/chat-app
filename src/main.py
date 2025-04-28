from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.api.endpoints import router
from src.config import MONGODB_DB_NAME, MONGODB_URI
from src.utils.db_services import connect_to_mongo

app = FastAPI(title="Ebla ChatBot Assistant")
app.include_router(router)

db = connect_to_mongo(MONGODB_URI, MONGODB_DB_NAME)
BASE_DIR = Path(__file__).parent
# STATIC_DIR = BASE_DIR / "static"
# TEMPLATES_DIR = BASE_DIR / "templates"

# Mount static files (CSS, images)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
print(BASE_DIR)
# Setup templates

# Allow CORS for frontend applications
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
