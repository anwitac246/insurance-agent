from pinecone import Pinecone
from app.core.config import settings

class VectorStore:
    pc: Pinecone = None
    
vs = VectorStore()

def connect_to_pinecone():
    if settings.PINECONE_API_KEY:
        vs.pc = Pinecone(api_key=settings.PINECONE_API_KEY)

def get_pinecone_index(index_name: str):
    if not vs.pc:
        return None
    return vs.pc.Index(index_name)
