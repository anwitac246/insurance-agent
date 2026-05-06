"""
decision_agent.py
-----------------
Final claims adjudicator with Groq key rotation support.
"""

from __future__ import annotations

import logging
import time

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.tools.groq_client import get_llm, record_429, record_success

logger = logging.getLogger(__name__)

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


def _invoke(inputs: dict) -> DecisionOutput:
    last_exc = None
    for attempt in range(LLM_MAX_RETRIES):
        try:
            llm = get_llm()
            chain = _prompt | llm.with_structured_output(DecisionOutput)
            result = chain.invoke(inputs)
            record_success()
            return result
        except Exception as exc:
            last_exc = exc
            exc_str = str(exc).lower()
            if "429" in exc_str or "rate limit" in exc_str or "rate_limit" in exc_str:
                logger.warning("decision_agent | 429 detected (attempt %d)", attempt + 1)
                record_429()
                time.sleep(2 ** attempt)
            else:
                logger.warning("decision_agent | LLM error (attempt %d): %s", attempt + 1, exc)
                time.sleep(2 ** attempt)
    raise last_exc


def run_decision_agent(state: dict) -> dict:
    t0 = time.perf_counter()
    errors: list[str] = list(state.get("errors", []))
    sanitized = state.get("sanitized_data", {})
    policy_verdict = state.get("policy_verdict", {})
    fraud_report = state.get("fraud_report", {})

    try:
        result: DecisionOutput = _invoke({
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