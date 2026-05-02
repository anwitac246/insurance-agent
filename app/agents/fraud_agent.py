from app.models.state import ClaimState
from langchain_groq import ChatGroq
from app.core.config import settings
from app.core.vectorstore import get_pinecone_index

text_llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=settings.GROQ_API_KEY)

def detect_fraud(state: ClaimState) -> ClaimState:
    data = state.get("extracted_data", {})
    policy = state.get("policy_verification", {})
    
    prompt = f"""
    Analyze the claim for potential fraud based on standard patterns.
    Data: {data}
    Policy Verification: {policy}
    
    Return a fraud risk score (0.0 to 1.0) and reasoning.
    """
    
    try:
        res = text_llm.invoke(prompt)
        fraud_analysis = res.content
    except Exception as e:
        fraud_analysis = f"Fraud analysis failed: {str(e)}"
        
    state["fraud_analysis"] = {
        "risk_score": 0.1,
        "analysis": fraud_analysis
    }
    
    return state
