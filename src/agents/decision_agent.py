from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic


class DecisionOutput(BaseModel):
    approved: bool
    final_payout: float = Field(description="Calculated payout after deductible and limit checks")
    denial_reason: str = Field(default="")
    step_by_step_reasoning: str = Field(
        description="Full chain-of-thought: loss amount, deductible subtraction, limit cap, fraud/exclusion adjustments"
    )


_llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0)

_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are the final claims adjudicator for a car insurance company. "
        "Your payout formula is: Final_Payout = min(Estimated_Loss - Deductible, Remaining_Limit). "
        "If errors exist, the claim is denied and payout is 0. "
        "If fraud risk is High, the claim is denied. "
        "If an exclusion is triggered, the claim is denied. "
        "Provide complete step-by-step reasoning showing every calculation.",
    ),
    (
        "human",
        "=== Claim Summary ===\n"
        "Claim ID: {claim_id}\n"
        "Estimated Loss: ${estimated_loss}\n\n"
        "=== Policy Verdict ===\n"
        "{policy_verdict}\n\n"
        "=== Fraud Report ===\n"
        "{fraud_report}\n\n"
        "=== Errors / Flags ===\n"
        "{errors}\n\n"
        "Calculate the final payout and provide a complete adjudication decision.",
    ),
])

_chain = _prompt | _llm.with_structured_output(DecisionOutput)


def run_decision_agent(state: dict) -> dict:
    errors: list[str] = list(state.get("errors", []))
    sanitized = state.get("sanitized_data", {})
    policy_verdict = state.get("policy_verdict", {})
    fraud_report = state.get("fraud_report", {})

    try:
        result: DecisionOutput = _chain.invoke({
            "claim_id": state.get("claim_id"),
            "estimated_loss": sanitized.get("estimated_loss", 0),
            "policy_verdict": str(policy_verdict),
            "fraud_report": str(fraud_report),
            "errors": errors if errors else "None",
        })
    except Exception as exc:
        return {
            **state,
            "errors": errors + [f"Decision agent failed: {exc}"],
            "final_payout": 0.0,
            "final_decision": {"approved": False, "reason": str(exc)},
        }

    return {
        **state,
        "final_payout": result.final_payout,
        "final_decision": result.model_dump(),
    }