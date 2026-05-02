from app.models.state import ClaimState
from langchain_groq import ChatGroq
from app.core.config import settings

text_llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=settings.GROQ_API_KEY)

def make_decision(state: ClaimState) -> ClaimState:
    data = state.get("extracted_data", {})
    policy = state.get("policy_verification", {})
    fraud = state.get("fraud_analysis", {})
    
    prompt = f"""
    You are the final decision agent for a car insurance claim.
    Review all inputs and make a final decision (Approve, Reject, or Escalate to Human).
    
    Extracted Data: {data}
    Policy Status: {policy}
    Fraud Analysis: {fraud}
    
    Output the final status and reasoning.
    """
    
    try:
        res = text_llm.invoke(prompt)
        decision = res.content
    except Exception as e:
        decision = f"Decision failed: {str(e)}"
        
    risk_score = fraud.get("risk_score", 1.0)
    
    state["decision"] = {
        "final_status": "Approved" if risk_score < 0.5 else "Escalate",
        "reasoning": decision
    }
    state["status"] = "completed"
    
    return state
