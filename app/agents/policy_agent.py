from app.models.state import ClaimState
from langchain_groq import ChatGroq
from app.core.config import settings
from app.core.vectorstore import get_pinecone_index

text_llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=settings.GROQ_API_KEY)

def verify_policy(state: ClaimState) -> ClaimState:
    extracted = state.get("extracted_data", {})
    
    prompt = f"""
    Verify the policy based on the extracted data.
    Data: {extracted}
    Assuming standard comprehensive coverage, does this claim appear valid?
    Identify any obvious exclusions.
    """
    
    try:
        res = text_llm.invoke(prompt)
        verification = res.content
    except Exception as e:
        verification = f"Verification failed: {str(e)}"
        
    state["policy_verification"] = {
        "is_valid": True,
        "details": verification
    }
    
    return state
