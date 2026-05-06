from langgraph.graph import StateGraph, END
from src.graph.state import ClaimState
from src.agents.verification_agent import run_verification_agent
from src.agents.policy_agent import run_policy_agent
from src.agents.fraud_agent import run_fraud_agent
from src.agents.decision_agent import run_decision_agent


def should_continue(state: ClaimState) -> str:
    return "failure_node" if state.get("errors") else "policy_validation"


def failure_node(state: ClaimState) -> ClaimState:
    return {
        **state,
        "final_payout": 0.0,
        "final_decision": {
            "approved": False,
            "denial_reason": "Claim failed verification checks.",
            "errors": state["errors"],
            "step_by_step_reasoning": (
                "Claim was rejected at the Document Verification stage. "
                f"Errors: {'; '.join(state['errors'])}"
            ),
        },
    }


def build_graph() -> StateGraph:
    graph = StateGraph(ClaimState)

    graph.add_node("document_verification", run_verification_agent)
    graph.add_node("failure_node", failure_node)
    graph.add_node("policy_validation", run_policy_agent)
    graph.add_node("fraud_detection", run_fraud_agent)
    graph.add_node("decision", run_decision_agent)

    graph.set_entry_point("document_verification")

    graph.add_conditional_edges(
        "document_verification",
        should_continue,
        {
            "failure_node": "failure_node",
            "policy_validation": "policy_validation",
        },
    )

    graph.add_edge("failure_node", END)
    graph.add_edge("policy_validation", "fraud_detection")
    graph.add_edge("fraud_detection", "decision")
    graph.add_edge("decision", END)

    return graph


compiled_graph = build_graph().compile()