"""
decision_agent.py
-----------------
Final claims adjudicator — async-first.

v2 changes
----------
- REMOVED the two-call approach (structured call + plain-text call to parse
  `approved`). This was the source of both the 400 Bad Request errors AND
  double LLM usage per claim.

  Now uses a SINGLE structured-output call where `approved` is typed as `str`
  (like the boolean fields in fraud_agent and policy_agent). Pydantic v2 coerces
  any JSON value — True, False, "yes", "no" — to str without raising, which
  eliminates the Groq tool_use_failed 400s caused by Literal["yes","no"]
  constraints.

  LLM call count per claim: 2 → 1

- Fixed latent NameError: `Optional` is now imported at the top of the file
  before it is used in type annotations. In Python < 3.10, `from __future__
  import annotations` defers evaluation but any explicit runtime reference
  (e.g., function signatures not under annotations) would still fail.

- Removed the dead _DecisionOutputRaw and _parse_approved helpers — no longer
  needed and were a maintenance hazard.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional  # must be imported before first use

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.tools.groq_client import get_async_llm, record_429, record_success

logger = logging.getLogger(__name__)

LLM_MAX_RETRIES = 3


# ── Schemas ────────────────────────────────────────────────────────────────────

class _DecisionOutputRaw(BaseModel):
    """
    Raw LLM output schema.

    `approved` typed as `str` — Pydantic v2 coerces True/False/"yes"/"no" to str
    without raising, eliminating Groq 400 Bad Request tool_use_failed errors that
    occur when the model outputs JSON True instead of the string "yes".
    """
    approved: str = Field(
        description=(
            'Must be the exact string "yes" or "no". '
            "NEVER output True, False, true, or false."
        )
    )
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


# Public schema — proper Python bool, used by downstream metrics and API response
class DecisionOutput(BaseModel):
    approved: bool
    final_payout: float
    denial_reason: str = ""
    step_by_step_reasoning: str


def _parse_bool(val: str) -> bool:
    return str(val).lower().strip() in ("yes", "true", "1")


# ── Prompt ─────────────────────────────────────────────────────────────────────

_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are the final claims adjudicator for a car insurance company.\n"
        "Payout formula: Final_Payout = min(Estimated_Loss - Deductible, Remaining_Limit)\n"
        "Deny (approved='no', payout=0) if ANY of:\n"
        "  - errors list is non-empty\n"
        "  - fraud_report.risk_score == 'High'\n"
        "  - policy_verdict.exclusion_triggered is true\n"
        "  - policy_verdict.incident_covered is false\n\n"
        "Provide complete step-by-step reasoning showing every calculation.\n\n"
        "CRITICAL — OUTPUT FORMAT:\n"
        '  approved MUST be the exact string "yes" or "no".\n'
        "  NEVER output True, False, true, or false for this field.\n"
        '  Correct: approved: "yes"   or   approved: "no"\n'
        "  WRONG:   approved: True    ← causes an API error\n",
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
        'Calculate the final payout. approved must be exactly "yes" or "no".',
    ),
])


# ── Async LLM invocation — single call ────────────────────────────────────────

async def _ainvoke(inputs: dict) -> DecisionOutput:
    """
    Single structured-output call.
    `approved` is a str field — Pydantic v2 accepts True/False/"yes"/"no" all coerced to str,
    avoiding the Groq 400 tool_use_failed error that occurred with Literal["yes","no"].
    """
    last_exc: Optional[Exception] = None

    for attempt in range(LLM_MAX_RETRIES):
        try:
            llm = get_async_llm()
            chain = _prompt | llm.with_structured_output(_DecisionOutputRaw)
            raw: _DecisionOutputRaw = await chain.ainvoke(inputs)
            record_success()

            return DecisionOutput(
                approved=_parse_bool(raw.approved),
                final_payout=raw.final_payout,
                denial_reason=raw.denial_reason,
                step_by_step_reasoning=raw.step_by_step_reasoning,
            )

        except Exception as exc:
            last_exc = exc
            exc_str = str(exc).lower()
            if "400" in exc_str or "bad request" in exc_str:
                logger.error(
                    "decision_agent | 400 Bad Request (non-retryable): %s", exc
                )
                raise
            if "429" in exc_str or "rate limit" in exc_str or "rate_limit" in exc_str:
                logger.warning(
                    "decision_agent | 429 detected (attempt %d)", attempt + 1
                )
                record_429()
                await asyncio.sleep(2 ** attempt)
            else:
                logger.warning(
                    "decision_agent | LLM error (attempt %d): %s", attempt + 1, exc
                )
                await asyncio.sleep(2 ** attempt)

    raise last_exc  # type: ignore[misc]


# ── Async main ─────────────────────────────────────────────────────────────────

async def arun_decision_agent(state: dict) -> dict:
    t0 = time.perf_counter()
    errors: list[str] = list(state.get("errors", []))
    sanitized = state.get("sanitized_data", {})
    policy_verdict = state.get("policy_verdict", {})
    fraud_report = state.get("fraud_report", {})

    estimated_loss = float(sanitized.get("estimated_loss", 0))

    try:
        result: DecisionOutput = await _ainvoke({
            "claim_id": state.get("claim_id"),
            "estimated_loss": f"{estimated_loss:,.2f}",
            "policy_verdict": str(policy_verdict),
            "fraud_report": str(fraud_report),
            "errors": errors if errors else "None",
        })
    except Exception as exc:
        logger.exception(
            "Decision agent failed for claim %s", state.get("claim_id")
        )
        return {
            **state,
            "errors": errors + [f"Decision agent failed: {exc}"],
            "final_payout": 0.0,
            "final_decision": {
                "approved": False,
                "denial_reason": str(exc),
                "step_by_step_reasoning": "",
            },
        }

    elapsed = time.perf_counter() - t0
    logger.info(
        "decision_agent | claim=%s | approved=%s | payout=%.2f | %.2fs (1 LLM call)",
        state.get("claim_id"), result.approved, result.final_payout, elapsed,
    )

    return {
        **state,
        "final_payout": result.final_payout,
        "final_decision": result.model_dump(),
    }


# ── Sync wrapper ───────────────────────────────────────────────────────────────

def run_decision_agent(state: dict) -> dict:
    return asyncio.run(arun_decision_agent(state))