from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
import uuid
import os
import shutil
import asyncio
from typing import List
from app.core.config import settings
from app.models.schemas import ClaimCreate, ClaimResponse
from app.core.database import get_database
from app.models.state import ClaimState

router = APIRouter()

@router.post("/", response_model=ClaimResponse)
async def create_claim(claim: ClaimCreate):
    claim_id = f"CLM-{uuid.uuid4().hex[:8].upper()}"
    
    db = get_database()
    claim_doc = {
        "claim_id": claim_id,
        "status": "pending_docs",
        "user_id": claim.user_id,
        "local_files": {"documents": [], "images": []},
        "extracted_data": {},
        "policy_verification": {},
        "fraud_analysis": {},
        "decision": {},
        "missing_docs": []
    }
    
    await db["claims"].insert_one(claim_doc)
    
    claim_dir = os.path.join(settings.DATA_DIR, "claims", claim_id)
    os.makedirs(os.path.join(claim_dir, "documents"), exist_ok=True)
    os.makedirs(os.path.join(claim_dir, "images"), exist_ok=True)
    
    return ClaimResponse(claim_id=claim_id, status="pending_docs", message="Claim created. Please upload documents.")

@router.post("/{claim_id}/upload")
async def upload_document(claim_id: str, files: List[UploadFile] = File(...)):
    db = get_database()
    claim = await db["claims"].find_one({"claim_id": claim_id})
    if not claim:
        raise HTTPException(status_code=404, detail="Claim not found")
        
    claim_dir = os.path.join(settings.DATA_DIR, "claims", claim_id)
    
    saved_docs = []
    saved_images = []
    
    for file in files:
        ext = file.filename.split('.')[-1].lower()
        if ext in ['jpg', 'jpeg', 'png', 'mp4']:
            folder = "images"
            filepath = os.path.join(claim_dir, folder, file.filename)
            saved_images.append(filepath)
        else:
            folder = "documents"
            filepath = os.path.join(claim_dir, folder, file.filename)
            saved_docs.append(filepath)
            
        with open(filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
    await db["claims"].update_one(
        {"claim_id": claim_id},
        {"$push": {
            "local_files.documents": {"$each": saved_docs},
            "local_files.images": {"$each": saved_images}
        }}
    )
    
    return {"message": f"Successfully uploaded {len(files)} files", "claim_id": claim_id}

# Synchronous execution helper for background task
def run_orchestrator(state_dict):
    from app.agents.orchestrator import orchestrator_app
    
    # Note: we exclude the MongoDB '_id' field for state compatibility
    if '_id' in state_dict:
        del state_dict['_id']
        
    result_state = orchestrator_app.invoke(state_dict)
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(update_claim_in_db(state_dict['claim_id'], result_state))
    loop.close()

async def update_claim_in_db(claim_id: str, result_state: dict):
    db = get_database()
    await db["claims"].update_one(
        {"claim_id": claim_id},
        {"$set": result_state}
    )

@router.post("/{claim_id}/process")
async def process_claim(claim_id: str, background_tasks: BackgroundTasks):
    db = get_database()
    claim = await db["claims"].find_one({"claim_id": claim_id})
    if not claim:
        raise HTTPException(status_code=404, detail="Claim not found")
        
    background_tasks.add_task(run_orchestrator, claim)
    
    return {"message": "Processing started in background", "claim_id": claim_id}
    
@router.get("/{claim_id}")
async def get_claim(claim_id: str):
    db = get_database()
    claim = await db["claims"].find_one({"claim_id": claim_id})
    if not claim:
        raise HTTPException(status_code=404, detail="Claim not found")
    
    claim["_id"] = str(claim["_id"])
    return claim
