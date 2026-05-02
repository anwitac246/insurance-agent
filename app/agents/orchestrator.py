from langgraph.graph import StateGraph, END
from app.models.state import ClaimState
from app.agents.document_agent import process_documents
from app.agents.policy_agent import verify_policy
from app.agents.fraud_agent import detect_fraud
from app.agents.decision_agent import make_decision

def should_continue(state: ClaimState) -> str:
    if state.get("status") == "pending_docs":
        return "end"
    return "verify_policy"

def build_graph():
    workflow = StateGraph(ClaimState)
    
    workflow.add_node("process_documents", process_documents)
    workflow.add_node("verify_policy", verify_policy)
    workflow.add_node("detect_fraud", detect_fraud)
    workflow.add_node("make_decision", make_decision)
    
    workflow.set_entry_point("process_documents")
    
    workflow.add_conditional_edges(
        "process_documents",
        should_continue,
        {
            "verify_policy": "verify_policy",
            "end": END
        }
    )
    
    workflow.add_edge("verify_policy", "detect_fraud")
    workflow.add_edge("detect_fraud", "make_decision")
    workflow.add_edge("make_decision", END)
    
    app = workflow.compile()
    return app

orchestrator_app = build_graph()
