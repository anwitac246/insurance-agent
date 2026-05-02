from motor.motor_asyncio import AsyncIOMotorClient
from app.core.config import settings

class Database:
    client: AsyncIOMotorClient = None

db = Database()

async def connect_to_mongo():
    db.client = AsyncIOMotorClient(settings.MONGODB_URI)
    
async def close_mongo_connection():
    if db.client:
        db.client.close()
    
def get_database():
    return db.client[settings.DATABASE_NAME]
