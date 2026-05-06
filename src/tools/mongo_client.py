import os
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.database import Database
from pinecone import Pinecone

load_dotenv()

_mongo_client: MongoClient | None = None
_pinecone_index = None


def get_db() -> Database:
    global _mongo_client
    if _mongo_client is None:
        _mongo_client = MongoClient(os.getenv("MONGODB_URL"))
    return _mongo_client["car_insurance_mas"]


def get_pinecone_index():
    global _pinecone_index
    if _pinecone_index is None:
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        _pinecone_index = pc.Index("insurance-policies")
    return _pinecone_index