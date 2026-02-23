from pymongo import MongoClient
from dotenv import load_dotenv
import os
from src.exception.exception import CustomException
from src.logging.logging import logging

load_dotenv()

URI = os.getenv("MONGO_DB_URI")

MONGO_URI = URI

try:
    logging.info("connecting to mongodb")
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    
    client.admin.command("ping")
    
    print("✅ MongoDB connection successful!")
    logging.info("connection successful")

except Exception as e:
    logging.error("Conection failed")
    print("Connection failed")
    raise CustomException(e)