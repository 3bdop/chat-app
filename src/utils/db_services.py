import logging

from pymongo import MongoClient


def connect_to_mongo(uri: str, db_name: str):
    """
    Connect to the MongoDB database using the provided URI and database name.
    """
    try:
        client = MongoClient(uri)
        db = client[db_name]
        logging.info(f"Successfully connected to MongoDB database: {db_name}")
        return db
    except Exception as e:
        logging.error("Error connecting to MongoDB: %s", e)
        raise
