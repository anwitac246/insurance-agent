"""
nma_agent.py
------------
Single-agent insurance claim adjudicator.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from nma_src.schemas.nma_schema import NMAOutput
from src.tools.llm_client import get_async_llm, record_429, record_success

logger = logging.getLogger(__name__)

# ── Fraud signal constants (mirror MAS fraud_agent) ───────────────────────────
COLLUSION_SHOP = "Apex AutoBody & Collision"
FREQUENT_CLAIM_THRESHOLD = 3


# ── Raw LLM output schema ──────────────────────────────────────────────────────

class _NMAOutputRaw(BaseModel):
    fraud_risk_score: str = Field(
        description=(
            "Overall fraud risk score as a number between 1 and 10. "
            "Score 8-10 for High risk (collusion, staging, frequent fraud history). "
            "Score 4-7 for Medium risk. Score 1-3 for Low risk. "
            "Output as a plain number string, e.g. '8'."
        ),
    )
    fraud_anomalies: str = Field(
        description="Comma-separated list of detected anomalies, or 'None'."
    )
    incident_covered: str = Field(
        description=(
            "Output 'yes' if the incident type is covered by coverage_scope "
            "and no exclusion applies. Output 'no' otherwise."
        )
    )
    exclusion_triggered: str = Field(
        description=(
            "Output 'yes' if a specific policy exclusion clause directly applies "
            "to the incident narrative. Output 'no' otherwise."
        )
    )
    exclusion_reason: str = Field(
        description="Exact quote of the triggered exclusion clause, or empty string."
    )
    final_payout: str = Field(
        description=(
            "Calculated payout as a decimal string using the formula: "
            "min(estimated_loss - deductible, remaining_limit). "
            "Set to '0.0' if claim is denied. Example: '4250.00'"
        )
    )
    approved: str = Field(
        description=(
            'Output the exact string "yes" if the claim is approved, '
            '"no" if denied. NEVER output True or False.'
        )
    )
    step_by_step_reasoning: str = Field(
        description="ONE sentence explaining the final decision."
    )


def _parse_bool(val) -> bool:
    return str(val).lower().strip() in ("yes", "true", "1")


# ── Prompt ─────────────────────────────────────────────────────────────────────

_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a senior insurance claims adjudicator responsible for the entire "
        "end-to-end assessment of a car insurance claim in a single pass.\n\n"
        "You must simultaneously perform:\n"
        "  1. FRAUD DETECTION — analyse all signals (claim history, repair shop, "
        "     narrative vs estimate mismatch, semantic exclusion matching)\n"
        "  2. POLICY COVERAGE CHECK — verify the incident is covered and no exclusion applies\n"
        "  3. FINANCIAL CALCULATION — payout = min(estimated_loss - deductible, remaining_limit)\n\n"
        "DENIAL RULES (deny if ANY of these are true):\n"
        "  • frequent_claims_signal is YES (customer has >= 3 denied/flagged prior claims)\n"
        "  • aggregate_breach_signal is YES (loss exceeds remaining policy limit)\n"
        "  • collusion_signal is YES (repair shop is in fraud network)\n"
        "  • staging_signal is YES (catastrophic narrative but tiny repair estimate)\n"
        "  • A policy exclusion clause directly and verbatim applies to the narrative\n"
        "  • The incident type is not covered by the policy coverage scope\n\n"
        "PAYOUT FORMULA: min(estimated_loss - deductible, remaining_limit)\n"
        "Set payout to 0.0 for any denied claim.\n\n"
        "CRITICAL OUTPUT FORMAT:\n"
        '  approved must be the exact string "yes" or "no". NEVER True or False.\n'
        "  final_payout must be a decimal string like '4250.00', not a number.\n"
        "  Keep step_by_step_reasoning to exactly ONE SHORT SENTENCE.",
    ),
    (
        "human",
        "=== CLAIM ===\n"
        "Claim ID        : {claim_id}\n"
        "Incident Type   : {incident_type}\n"
        "Narrative       : {narrative}\n"
        "Estimated Loss  : ${estimated_loss}\n"
        "Repair Shop     : {repair_shop}\n"
        "OCR Estimate    : ${ocr_estimate}\n\n"
        "=== PRE-COMPUTED FRAUD SIGNALS ===\n"
        "(These are deterministic — do NOT override them)\n"
        "Frequent Claims Signal (>={freq_threshold} denied/flagged): {frequent_claims_signal}\n"
        "Aggregate Breach Signal (loss > remaining limit)          : {aggregate_breach_signal}\n"
        "Collusion Shop Signal (flagged repair network)            : {collusion_signal}\n"
        "Staging Signal (high severity narrative, tiny estimate)   : {staging_signal}\n\n"
        "=== CLAIM HISTORY ===\n"
        "{history_summary}\n\n"
        "=== CUSTOMER PROFILE ===\n"
        "Risk Rating : {risk_rating}\n"
        "NCD Tier    : {ncd_tier}\n\n"
        "=== POLICY ===\n"
        "Policy ID               : {policy_id}\n"
        "Coverage Scope          : {coverage_scope}\n"
        "Exclusions              : {exclusions}\n"
        "Policy Limit            : ${policy_limit}\n"
        "Aggregate Limit         : ${aggregate_limit}\n"
        "Deductible              : ${deductible}\n"
        "Total Historical Payout : ${total_historical_payout}\n"
        "Remaining Limit         : ${remaining_limit}\n\n"
        "Using ALL signals above, adjudicate the claim. "
        'approved must be exactly "yes" or "no".',
    ),
])


# ── Deterministic pre-compute helpers ─────────────────────────────────────────

def _compute_signals(context: dict) -> dict:
    """
    Compute deterministic fraud and financial signals before the LLM call.
    These mirror the MAS verification/fraud/policy agents and are injected
    explicitly into the prompt so the LLM doesn't have to infer them.
    """
    claim: dict = context.get("claim", {})
    customer: dict = context.get("customer", {})
    policy: dict = context.get("policy", {})
    history: list[dict] = context.get("history_records", [])

    # Claim fields
    estimated_loss = float(claim.get("estimated_loss", 0))
    repair_shop: str = claim.get("ocr_extraction", {}).get("RepairShopName", "")
    ocr_estimate = float(claim.get("ocr_extraction", {}).get("TotalEstimate", estimated_loss))

    # Customer fields
    risk_rating: str = customer.get("risk_rating", "Unknown")
    ncd_tier: float = float(customer.get("ncd_tier", 0.0))

    # Policy fields
    aggregate_limit = float(policy.get("aggregate_limit", 0))
    total_historical_payout = float(policy.get("total_historical_payout", 0))
    remaining_limit = aggregate_limit - total_historical_payout
    policy_limit = float(policy.get("policy_limit", 0))
    deductible = float(policy.get("deductible", 0))

    # Signal: frequent claimant
    denied_flagged = sum(
        1 for r in history if r.get("claim_status") in ("Denied", "Fraud_Flagged")
    )
    frequent_claims_signal = denied_flagged >= FREQUENT_CLAIM_THRESHOLD

    # Signal: aggregate limit breach
    aggregate_breach_signal = estimated_loss > remaining_limit

    # Signal: collusion ring (flagged repair shop)
    collusion_signal = COLLUSION_SHOP.lower() in repair_shop.lower()

    # Signal: staged accident (high narrative loss, tiny OCR estimate)
    staging_signal = ocr_estimate < 1000 and estimated_loss >= 5000

    return {
        "claim_id": claim.get("claim_id", ""),
        "incident_type": claim.get("incident_type", ""),
        "narrative": claim.get("narrative", ""),
        "estimated_loss": f"{estimated_loss:,.2f}",
        "repair_shop": repair_shop or "Not specified",
        "ocr_estimate": f"{ocr_estimate:,.2f}",
        "freq_threshold": FREQUENT_CLAIM_THRESHOLD,
        "frequent_claims_signal": "YES" if frequent_claims_signal else "NO",
        "aggregate_breach_signal": "YES" if aggregate_breach_signal else "NO",
        "collusion_signal": "YES" if collusion_signal else "NO",
        "staging_signal": "YES" if staging_signal else "NO",
        "history_summary": context.get("history_summary", "No history available."),
        "risk_rating": risk_rating,
        "ncd_tier": ncd_tier,
        "policy_id": policy.get("policy_id", "N/A"),
        "coverage_scope": policy.get("coverage_scope", "Not retrieved"),
        "exclusions": policy.get("exclusions", "Not retrieved"),
        "policy_limit": f"{policy_limit:,.2f}",
        "aggregate_limit": f"{aggregate_limit:,.2f}",
        "deductible": f"{deductible:,.2f}",
        "total_historical_payout": f"{total_historical_payout:,.2f}",
        "remaining_limit": f"{remaining_limit:,.2f}",
        # Keep raw values for deterministic post-processing
        "_frequent_claims_signal": frequent_claims_signal,
        "_aggregate_breach_signal": aggregate_breach_signal,
        "_collusion_signal": collusion_signal,
        "_staging_signal": staging_signal,
        "_remaining_limit": remaining_limit,
        "_deductible": deductible,
        "_estimated_loss": estimated_loss,
    }


# ── Async agent ────────────────────────────────────────────────────────────────

async def arun_nma_agent(context: dict) -> NMAOutput:
    """
    Single LLM call that adjudicates the entire claim.

    Deterministic signals are pre-computed and injected into the prompt,
    then used again post-LLM to enforce denial logic — the LLM cannot
    accidentally approve a claim that should be deterministically denied.
    """
    signals = _compute_signals(context)

    for attempt in range(3):
        try:
            llm = get_async_llm()
            chain = _PROMPT | llm.with_structured_output(_NMAOutputRaw)
            raw: _NMAOutputRaw = await chain.ainvoke(signals)
            record_success()

            # ── Deterministic post-processing overrides ────────────────────────
            # If any hard-denial signal fired, force denial regardless of LLM output.
            # This prevents the LLM from approving structurally invalid claims.
            must_deny = (
                signals["_frequent_claims_signal"]
                or signals["_aggregate_breach_signal"]
                or signals["_collusion_signal"]
                or signals["_staging_signal"]
            )

            approved_llm = _parse_bool(raw.approved)
            approved_final = approved_llm and not must_deny

            try:
                payout_raw = float(raw.final_payout)
            except (ValueError, TypeError):
                payout_raw = 0.0

            final_payout = payout_raw if approved_final else 0.0

            try:
                fraud_score = max(1, min(10, int(float(raw.fraud_risk_score))))
            except (ValueError, TypeError):
                fraud_score = 5  # neutral fallback

            return NMAOutput(
                fraud_risk_score=fraud_score,
                fraud_anomalies=raw.fraud_anomalies,
                incident_covered=_parse_bool(raw.incident_covered),
                exclusion_triggered=_parse_bool(raw.exclusion_triggered),
                exclusion_reason=raw.exclusion_reason,
                final_payout=final_payout,
                approved=approved_final,
                step_by_step_reasoning=raw.step_by_step_reasoning,
            )

        except Exception as exc:
            exc_str = str(exc).lower()
            if "429" in exc_str or "rate limit" in exc_str:
                record_429()
                await asyncio.sleep(2 ** attempt)
            elif "400" in exc_str:
                raise
            else:
                await asyncio.sleep(2 ** attempt)

    raise RuntimeError("NMA Agent failed after 3 retries")