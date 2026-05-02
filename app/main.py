from fastapi import FastAPI
from contextlib import asynccontextmanager
import os
from app.core.config import settings
from app.api import claims
from app.core.database import connect_to_mongo, close_mongo_connection
from app.core.vectorstore import connect_to_pinecone

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    os.makedirs(settings.DATA_DIR, exist_ok=True)
    os.makedirs(os.path.join(settings.DATA_DIR, "claims"), exist_ok=True)
    
    await connect_to_mongo()
    connect_to_pinecone()
    
    yield
    # Shutdown logic
    await close_mongo_connection()

app = FastAPI(title="Multi-Agent Claim Processing API", lifespan=lifespan)

app.include_router(claims.router, prefix="/claims", tags=["Claims"])

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
