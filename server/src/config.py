import logging
import os

from dotenv import load_dotenv

load_dotenv(override=True)

MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME")

# print(repr(MONGODB_URI))
# print(f"DB NAME {MONGODB_DB_NAME}")


if not MONGODB_URI:
    raise ValueError("MONGODB_URI is not set in .env")

if not MONGODB_DB_NAME:
    raise ValueError("MONGODB_DB_NAME is not set in .env")

logging.info(f"Successfully connected to MongoDB URI: {MONGODB_URI}")

logging.info(f"Successfully connected to MongoDB database: {MONGODB_DB_NAME}")
