"""
decision_agent.py
-----------------
Final claims adjudicator — async-first.

Fixes applied:
  - _DecisionOutputNoBools schema removes the `approved` boolean field entirely.
    Structured output is used only for the safe numeric/text fields; `approved`
    is parsed from a plain-text call via regex — eliminating tool_use_failed 400
    errors on Groq when the model outputs JSON True/False.
  - Field description on _DecisionOutputRaw.approved now explicitly forbids
    Python boolean literals (kept for reference, not used for LLM output).
  - System prompt includes a CRITICAL format block with correct/wrong examples.
  - 400 Bad Request errors still fail immediately (non-retryable).
  - Float values pre-formatted as strings to avoid LangChain template misparsing.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.tools.groq_client import get_async_llm, record_429, record_success

logger = logging.getLogger(__name__)

LLM_MAX_RETRIES = 3


# ── Schemas ────────────────────────────────────────────────────────────────────

# Structured output schema with the boolean `approved` field removed.
# Groq's model ignores Literal["yes","no"] constraints and outputs JSON True/False,
# causing 400 tool_use_failed. We parse `approved` from plain text instead.
class _DecisionOutputNoBools(BaseModel):
    """Structured output schema with approved field removed to avoid Groq 400s."""
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


# Kept for reference / legacy — not used for LLM output any more.
class _DecisionOutputRaw(BaseModel):
    approved: Literal["yes", "no"] = Field(
        description='Must be the exact string "yes" or the exact string "no". '
                    'NEVER output True, False, true, or false for this field.'
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


# Public schema (proper bool) used by downstream metrics and API response
class DecisionOutput(BaseModel):
    approved: bool
    final_payout: float
    denial_reason: str = ""
    step_by_step_reasoning: str


# ── Boolean parser ─────────────────────────────────────────────────────────────

_APPROVED_RE = re.compile(
    r'approved\s*[=:"\s]+\s*(yes|no|true|false)',
    re.IGNORECASE,
)


def _parse_approved(text: str) -> Optional[bool]:
    """Extract the approved flag from plain LLM text. Returns None if not found."""
    match = _APPROVED_RE.search(text)
    if match:
        return match.group(1).lower() in ("yes", "true")
    return None


# ── Prompt ─────────────────────────────────────────────────────────────────────

_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are the final claims adjudicator for a car insurance company. "
        "Your payout formula is: Final_Payout = min(Estimated_Loss - Deductible, Remaining_Limit). "
        "If errors exist, the claim is denied and payout is 0. "
        "If fraud risk is High, the claim is denied. "
        "If an exclusion is triggered, the claim is denied. "
        "Provide complete step-by-step reasoning showing every calculation.\n\n"
        "CRITICAL — OUTPUT FORMAT RULES:\n"
        '  The approved field MUST be the exact string "yes" or the exact string "no".\n'
        "  NEVER output True, False, true, or false for this field.\n"
        '  Correct:   approved: "yes"\n'
        '  Correct:   approved: "no"\n'
        "  WRONG:     approved: True    <- this will cause an API error\n"
        "  WRONG:     approved: false   <- this will cause an API error\n",
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
        "Calculate the final payout and provide a complete adjudication decision. "
        'Remember: approved must be exactly "yes" or "no".',
    ),
])


# ── Async LLM invocation ───────────────────────────────────────────────────────

# Type alias (Python 3.9 compat)
from typing import Optional


async def _ainvoke(inputs: dict) -> DecisionOutput:
    """
    Two-step invocation strategy:
      Step 1 — Structured output for safe fields (final_payout, denial_reason,
               step_by_step_reasoning). Boolean `approved` excluded from schema.
      Step 2 — Plain text call to extract `approved` via regex.
               Falls back to inferring approval from denial_reason if parsing fails.
    """
    last_exc = None

    for attempt in range(LLM_MAX_RETRIES):
        try:
            llm = get_async_llm()

            # ── Step 1: Structured output (no boolean fields) ──────────────────
            chain_struct = _prompt | llm.with_structured_output(_DecisionOutputNoBools)
            core: _DecisionOutputNoBools = await chain_struct.ainvoke(inputs)

            # ── Step 2: Plain text to extract `approved` ───────────────────────
            try:
                chain_text = _prompt | llm
                text_result = await chain_text.ainvoke(inputs)
                approved = _parse_approved(text_result.content)
                logger.debug(
                    "decision_agent | parsed approved=%s from plain text", approved
                )
            except Exception as bool_exc:
                logger.warning(
                    "decision_agent | plain-text approved extraction failed, "
                    "inferring from denial_reason: %s", bool_exc
                )
                approved = None

            # Fall back: if no `approved` parsed, infer from denial_reason + payout
            if approved is None:
                approved = not bool(core.denial_reason) and core.final_payout > 0

            record_success()
            return DecisionOutput(
                approved=approved,
                final_payout=core.final_payout,
                denial_reason=core.denial_reason,
                step_by_step_reasoning=core.step_by_step_reasoning,
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
                logger.warning("decision_agent | 429 detected (attempt %d)", attempt + 1)
                record_429()
                await asyncio.sleep(2 ** attempt)
            else:
                logger.warning(
                    "decision_agent | LLM error (attempt %d): %s", attempt + 1, exc
                )
                await asyncio.sleep(2 ** attempt)

    raise last_exc


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


def run_decision_agent(state: dict) -> dict:
    return asyncio.run(arun_decision_agent(state))