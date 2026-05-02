import os
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

class Settings(BaseSettings):
    GROQ_API_KEY: str = ""
    PINECONE_API_KEY: str = ""
    PINECONE_ENVIRONMENT: str = "us-east-1"
    MONGODB_URI: str = "mongodb://localhost:27017"
    DATABASE_NAME: str = "insurance_claims"
    DATA_DIR: str = "data"

    model_config = SettingsConfigDict(
        env_file=os.path.join(PROJECT_ROOT, ".env"), 
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
