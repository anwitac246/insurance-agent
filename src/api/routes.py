from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.main import process_claim

app = FastAPI(title="Car Insurance MAS API")


class ClaimRequest(BaseModel):
    claim_id: str


@app.post("/process-claim")
def process(req: ClaimRequest):
    try:
        result = process_claim(req.claim_id)
        return {
            "claim_id": result["claim_id"],
            "final_payout": result.get("final_payout"),
            "final_decision": result.get("final_decision"),
            "errors": result.get("errors"),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/health")
def health():
    return {"status": "ok"}