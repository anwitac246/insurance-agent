"""
decision_agent.py
-----------------
Final claims adjudicator.

Migrated from ChatAnthropic → ChatGroq (llama-3.3-70b-versatile).
No Anthropic API key is required anywhere in this project.
"""

from __future__ import annotations

import logging
import time
from functools import lru_cache

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

GROQ_MODEL = "llama-3.3-70b-versatile"
LLM_MAX_RETRIES = 3


class DecisionOutput(BaseModel):
    approved: bool
    final_payout: float = Field(
        description="Calculated payout after deductible and limit checks"
    )
    denial_reason: str = Field(default="")
    step_by_step_reasoning: str = Field(
        description=(
            "Full chain-of-thought: loss amount, deductible subtraction, "
            "limit cap, fraud/exclusion adjustments"
        )
    )


@lru_cache(maxsize=1)
def _get_llm() -> ChatGroq:
    return ChatGroq(model=GROQ_MODEL, temperature=0, request_timeout=45)


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


@retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(LLM_MAX_RETRIES),
    wait=wait_exponential(min=2, max=10),
    reraise=True,
)
def _invoke(chain, inputs: dict) -> DecisionOutput:
    return chain.invoke(inputs)


def run_decision_agent(state: dict) -> dict:
    t0 = time.perf_counter()
    errors: list[str] = list(state.get("errors", []))
    sanitized = state.get("sanitized_data", {})
    policy_verdict = state.get("policy_verdict", {})
    fraud_report = state.get("fraud_report", {})

    chain = _prompt | _get_llm().with_structured_output(DecisionOutput)

    try:
        result: DecisionOutput = _invoke(chain, {
            "claim_id": state.get("claim_id"),
            "estimated_loss": sanitized.get("estimated_loss", 0),
            "policy_verdict": str(policy_verdict),
            "fraud_report": str(fraud_report),
            "errors": errors if errors else "None",
        })
    except Exception as exc:
        logger.exception("Decision agent failed for claim %s", state.get("claim_id"))
        return {
            **state,
            "errors": errors + [f"Decision agent failed: {exc}"],
            "final_payout": 0.0,
            "final_decision": {"approved": False, "denial_reason": str(exc)},
        }

    elapsed = time.perf_counter() - t0
    logger.info(
        "decision_agent | claim=%s | approved=%s | payout=%.2f | %.2fs",
        state.get("claim_id"), result.approved, result.final_payout, elapsed,
    )

    return {
        **state,
        "final_payout": result.final_payout,
        "final_decision": result.model_dump(),
    }