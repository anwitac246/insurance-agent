import asyncio
from sentence_transformers import SentenceTransformer
from src.tools.mongo_client import get_db, get_pinecone_index

_embedder = None

def get_embedder() -> SentenceTransformer:
    global _embedder
    if not _embedder:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder

async def fetch_nma_context(claim_id: str) -> dict:
    db = get_db()
    claim = await asyncio.to_thread(db.claims.find_one, {"claim_id": claim_id})
    if not claim:
        raise ValueError(f"Claim {claim_id} not found.")

    cust_id = claim.get("customer_id")
    customer = await asyncio.to_thread(db.customers.find_one, {"customer_id": cust_id}) if cust_id else {}
    
    narrative = claim.get("incident_narrative", "")
    policy_id = claim.get("policy_id", "")
    
    policy_meta = {}
    if narrative and policy_id:
        vec = await asyncio.to_thread(get_embedder().encode, narrative)
        idx = get_pinecone_index()
        res = await asyncio.to_thread(
            idx.query,
            vector=vec.tolist(),
            top_k=1,
            filter={"policy_id": {"$eq": policy_id}},
            include_metadata=True
        )
        if res.get("matches"):
            policy_meta = res["matches"][0]["metadata"]
            
    return {
        "claim": claim,
        "customer": customer,
        "policy": policy_meta
    }
