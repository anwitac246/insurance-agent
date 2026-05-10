import sys
import os
import asyncio

# Setup sys.path so we can import from nma_src and the base repo src
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
nma_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
if nma_dir not in sys.path:
    sys.path.insert(0, nma_dir)

from nma_src.tools.context_fetcher import fetch_nma_context
from nma_src.agents.nma_agent import arun_nma_agent
from nma_src.schemas.nma_schema import NMAOutput

async def process_claim_nma(claim_id: str) -> NMAOutput:
    """End-to-end NMA process for a single claim."""
    print(f"Fetching context for {claim_id}...")
    context = await fetch_nma_context(claim_id)
    
    print(f"Calling Single Agent (LLM)...")
    result = await arun_nma_agent(context)
    
    return result

if __name__ == "__main__":
    cid = input("Enter Claim ID: ").strip()
    if cid:
        res = asyncio.run(process_claim_nma(cid))
        print("\n=== FINAL DECISION ===")
        print(f"Approved: {res.approved}")
        print(f"Payout: ${res.final_payout:,.2f}")
        print(f"Reasoning: {res.step_by_step_reasoning}")
        print(f"Fraud Risk: {res.fraud_risk_score}/10")
        if res.exclusion_triggered:
            print(f"Exclusion: {res.exclusion_reason}")
