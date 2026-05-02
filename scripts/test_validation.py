import asyncio
import os
from app.agents.document_agent import process_documents
from app.models.state import ClaimState

async def test_validation():
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    
    state: ClaimState = {
        "claim_id": "TEST-123",
        "status": "in_progress",
        "user_id": "usr-1",
        "local_files": {"documents": [], "images": []},
        "extracted_data": {},
        "policy_verification": {},
        "fraud_analysis": {},
        "decision": {},
        "missing_data": {"missing_documents": [], "missing_fields": []}
    }
    
    with open("data/dummy.txt", "w") as f:
        f.write("Policy Number: POL-9999\nClaimant Name: John Doe\nIncident Date: 2023-10-01\nVehicle Number: ABC-123\n")
        
    state["local_files"]["documents"].append("data/dummy.txt")
    
    print("Running process_documents...")
    result = process_documents(state)
    print("\n--- RESULTS ---")
    print("STATUS:", result.get("status"))
    print("MISSING DOCS:", result.get("missing_data", {}).get("missing_documents"))
    print("MISSING FIELDS:", result.get("missing_data", {}).get("missing_fields"))
    print("\nEXTRACTED DATA:")
    print(result.get("extracted_data"))

if __name__ == "__main__":
    asyncio.run(test_validation())
