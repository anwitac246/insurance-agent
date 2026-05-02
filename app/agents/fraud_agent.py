from app.models.state import ClaimState
from app.services.db_queries import get_policy_by_number, get_claims_by_vehicle, get_fraud_signals_by_entity
import asyncio
from langchain_groq import ChatGroq
from app.core.config import settings
from pydantic import BaseModel, Field
from typing import List
from datetime import datetime

class FraudAnalysisOutput(BaseModel):
    fraud_risk_score: float = Field(..., description="A score from 0.0 (no risk) to 1.0 (high risk)")
    fraud_flags: List[str] = Field(default_factory=list, description="List of specific fraud indicators found")
    requires_manual_review: bool = Field(..., description="True if risk is high or significant discrepancies exist")

text_llm = ChatGroq(model="llama-3.1-8b-instant", api_key=settings.GROQ_API_KEY)
structured_llm = text_llm.with_structured_output(FraudAnalysisOutput)

async def detect_fraud(state: ClaimState) -> ClaimState:
    extracted = state.get("extracted_data", {})
    policy_veri = state.get("policy_verification", {})
    
    policy_doc = extracted.get("insurance_policy", {})
    claim_form = extracted.get("claim_form", {})
    policy_num = (policy_doc.get("policy_number") or claim_form.get("policy_number") or "").strip()
    
    async def fetch_db_context():
        policy = await get_policy_by_number(policy_num) if policy_num else None
        past_claims = []
        signals = []
        if policy:
            veh_id = policy.get("vehicle_id")
            user_id = policy.get("policyholder_id")
            if veh_id:
                past_claims = await get_claims_by_vehicle(veh_id)
                signals.extend(await get_fraud_signals_by_entity(veh_id))
            if user_id:
                signals.extend(await get_fraud_signals_by_entity(user_id))
        return policy, past_claims, signals
        
    try:
        policy, past_claims, signals = await fetch_db_context()
        
        # Build synthesis prompt for the LLM
        prompt = f"""
        Analyze the following context for insurance fraud.
        
        Extracted Data: {extracted}
        Policy Verification Status: {policy_veri}
        
        Database Findings:
        - Past Claims on Vehicle: {len(past_claims)} claims. (Details: {[c.get('incident_date') for c in past_claims]})
        - Known Fraud Signals on User/Vehicle: {len(signals)} signals. (Details: {[s.get('fraud_type') for s in signals]})
        
        Determine the fraud risk score (0.0 to 1.0).
        Consider:
        - Are there mismatches in policy verification (e.g. wrong user, wrong vehicle)?
        - Does the vehicle have high claim frequency?
        - Are there known fraud signals?
        
        If there are any mismatches or fraud signals, the score should be > 0.5 and requires_manual_review should be true.
        """
        
        res = structured_llm.invoke(prompt)
        fraud_analysis = res.model_dump()
        
    except Exception as e:
        fraud_analysis = {
            "fraud_risk_score": 0.8, # fail-safe high risk on error
            "fraud_flags": [f"Analysis failed: {str(e)}"],
            "requires_manual_review": True
        }
        
    state["fraud_analysis"] = fraud_analysis
    
    return state
