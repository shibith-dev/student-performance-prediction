from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()

URI = os.getenv("MONGO_DB_URI")

MONGO_URI = URI

try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    
    client.admin.command("ping")
    
    print("✅ MongoDB connection successful!")

except Exception as e:
    print("❌ Connection failed:")
    print(e)