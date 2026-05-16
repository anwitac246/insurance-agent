"""
nma_agent.py
------------
Single-agent insurance claim adjudicator.

FIXES IN THIS VERSION
---------------------
1. EXCLUSION VERIFICATION — Added _exclusion_reason_verified() which checks that
   whatever exclusion the LLM quotes actually exists in the stored policy text.
   Previously the LLM could hallucinate exclusions like "Street Racing" onto a
   normal vandalism or rear-end claim and the system would blindly trust it.
   This was the PRIMARY driver of the 39% normal claim accuracy.

2. NO-SIGNAL OVERRIDE — If ALL four deterministic signals are NO and the LLM
   denies the claim WITHOUT a verified exclusion reason, the decision is
   overridden to Approved. An LLM running on llama3:latest with a massive
   context prompt regularly hallucinates denial reasons when no signal is
   active — this guard catches those cases.

3. frequent_claims_signal ADDED TO must_deny — Previously only aggregate_breach,
   collusion, and staging were hard overrides. frequent_claims_signal was left
   to the LLM which is non-deterministic. Now all four deterministic signals
   are enforced as hard overrides, making the NMA more consistent.

4. FRAUD SCORE CLAMPED when all signals are clear — Prevents the LLM from
   escalating fraud risk to HIGH on clean claims, which was causing the
   decision agent to deny them even when the approval flag was "yes".

5. PAYOUT RECALCULATION GUARD — If the claim is approved but the LLM returned
   a payout of 0.0 (a known failure mode on local models), the payout is
   deterministically recalculated as min(loss - deductible, remaining_limit).

6. PROMPT IMPROVED — Adds explicit ALL_SIGNALS_CLEAR indicator to the prompt
   and instructs the LLM that it MUST approve when that flag is YES and no
   exact exclusion clause can be quoted verbatim from the Exclusions list.

7. Collusion shop detection retains full-name substring match (case-insensitive)
   from the previous version.
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

COLLUSION_SHOP = "Apex AutoBody & Collision"
FREQUENT_CLAIM_THRESHOLD = 3


# ── Output schema ──────────────────────────────────────────────────────────────

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
            "Output 'yes' ONLY if a specific exclusion clause from the Exclusions "
            "list directly and unambiguously matches the incident narrative. "
            "Output 'no' otherwise. Do NOT invent exclusions."
        )
    )
    exclusion_reason: str = Field(
        description=(
            "If an exclusion applies, copy the EXACT phrase from the Exclusions "
            "list verbatim — do NOT paraphrase or rephrase. "
            "Leave as empty string if no exclusion applies."
        )
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


# ── Helpers ────────────────────────────────────────────────────────────────────

def _parse_bool(val) -> bool:
    return str(val).lower().strip() in ("yes", "true", "1")


def _parse_float(val: str) -> float:
    try:
        return float(str(val).replace("$", "").replace(",", "").strip())
    except ValueError:
        clean_val = "".join(c for c in str(val) if c.isdigit() or c in ".-")
        return float(clean_val) if clean_val else 0.0


def _exclusion_reason_verified(exclusion_reason: str, exclusions_text: str) -> bool:
    """
    FIX: Verify that the LLM's quoted exclusion_reason actually exists in the
    stored policy exclusions text. This prevents hallucinated exclusions from
    denying legitimate claims.

    Strategy: extract key words (>4 chars) from the quoted reason and check
    that at least 50% of them appear in the policy exclusions text. This is
    robust against minor punctuation differences between the LLM output and
    the stored text, while still catching outright hallucinations.

    Returns:
        True  — reason is verifiable → exclusion denial is legitimate
        False — reason cannot be found in policy text → do NOT trust LLM
    """
    if not exclusion_reason or not exclusion_reason.strip():
        return False
    if not exclusions_text:
        return False

    key_words = [
        w.strip(".,;:()\"'")
        for w in exclusion_reason.lower().split()
        if len(w.strip(".,;:()\"'")) > 4
    ]
    if not key_words:
        return False

    exclusions_lower = exclusions_text.lower()
    matches = sum(1 for w in key_words if w in exclusions_lower)
    verified = (matches / len(key_words)) >= 0.5

    if not verified:
        logger.debug(
            "Exclusion not verified: '%s' — %d/%d key words found in policy text.",
            exclusion_reason, matches, len(key_words),
        )
    return verified


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
        "  • A policy exclusion clause semantically matches the incident narrative\n"
        "  • The incident type is not covered by the policy coverage scope\n\n"
        "CRITICAL DEFAULT RULE — READ THIS CAREFULLY:\n"
        "  If ALL_SIGNALS_CLEAR is YES (all four fraud signals are NO), you MUST "
        "approve the claim UNLESS you can quote an exact clause from the Exclusions "
        "list that directly and unambiguously matches the incident narrative.\n"
        "  Standard incidents (rear-end collision, theft, vandalism, hail damage, "
        "flood, animal strike, hit-and-run) on a policy with matching coverage scope "
        "are APPROVED by default. Do NOT invent fraud signals or exclusions.\n"
        "  If you deny a claim when ALL_SIGNALS_CLEAR is YES, you MUST quote the "
        "exact exclusion text verbatim — otherwise approve.\n\n"
        "EXCLUSION RULE:\n"
        "  Only trigger an exclusion if the narrative clearly describes an activity "
        "explicitly listed in the Exclusions field. Quote the exact text. "
        "If the connection is ambiguous or indirect — do NOT trigger the exclusion.\n\n"
        "PAYOUT FORMULA: min(estimated_loss - deductible, remaining_limit)\n"
        "Set payout to 0.0 for any denied claim. Calculate correctly for approved claims.\n\n"
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
        "ALL_SIGNALS_CLEAR: {all_signals_clear}\n"
        "→ If YES and no exact exclusion applies: you MUST output approved='yes'\n\n"
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


# ── Signal pre-computation ─────────────────────────────────────────────────────

def _compute_signals(context: dict) -> dict:
    claim: dict = context.get("claim", {})
    customer: dict = context.get("customer", {})
    policy: dict = context.get("policy", {})
    history: list[dict] = context.get("history_records", [])

    estimated_loss = float(claim.get("estimated_loss", 0))
    repair_shop: str = claim.get("ocr_extraction", {}).get("RepairShopName", "")
    ocr_estimate = float(claim.get("ocr_extraction", {}).get("TotalEstimate", estimated_loss))

    risk_rating: str = customer.get("risk_rating", "Unknown")
    ncd_tier: float = float(customer.get("ncd_tier", 0.0))

    aggregate_limit = float(policy.get("aggregate_limit", 0))
    total_historical_payout = float(policy.get("total_historical_payout", 0))
    remaining_limit = aggregate_limit - total_historical_payout
    policy_limit = float(policy.get("policy_limit", 0))
    deductible = float(policy.get("deductible", 0))
    exclusions_text: str = policy.get("exclusions", "")

    denied_flagged = sum(
        1 for r in history if r.get("claim_status") in ("Denied", "Fraud_Flagged")
    )
    frequent_claims_signal = denied_flagged >= FREQUENT_CLAIM_THRESHOLD

    aggregate_breach_signal = estimated_loss > remaining_limit

    # Full-name substring match (case-insensitive) — more robust than first-word only
    collusion_signal = COLLUSION_SHOP.lower() in repair_shop.lower()

    staging_signal = ocr_estimate < 1000 and estimated_loss >= 5000

    # FIX: Explicit "all signals clear" flag passed to the LLM as a strong hint
    all_signals_clear = not (
        frequent_claims_signal
        or aggregate_breach_signal
        or collusion_signal
        or staging_signal
    )

    return {
        # ── Prompt-facing fields ───────────────────────────────────────────────
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
        "all_signals_clear": "YES" if all_signals_clear else "NO",
        "history_summary": context.get("history_summary", "No history available."),
        "risk_rating": risk_rating,
        "ncd_tier": ncd_tier,
        "policy_id": policy.get("policy_id", "N/A"),
        "coverage_scope": policy.get("coverage_scope", "Not retrieved"),
        "exclusions": exclusions_text,
        "policy_limit": f"{policy_limit:,.2f}",
        "aggregate_limit": f"{aggregate_limit:,.2f}",
        "deductible": f"{deductible:,.2f}",
        "total_historical_payout": f"{total_historical_payout:,.2f}",
        "remaining_limit": f"{remaining_limit:,.2f}",
        # ── Internal fields (prefixed with _ for post-processing only) ─────────
        "_frequent_claims_signal": frequent_claims_signal,
        "_aggregate_breach_signal": aggregate_breach_signal,
        "_collusion_signal": collusion_signal,
        "_staging_signal": staging_signal,
        "_all_signals_clear": all_signals_clear,
        "_remaining_limit": remaining_limit,
        "_deductible": deductible,
        "_estimated_loss": estimated_loss,
        "_exclusions_text": exclusions_text,
    }


# ── Async agent ────────────────────────────────────────────────────────────────

async def arun_nma_agent(context: dict) -> NMAOutput:
    signals = _compute_signals(context)

    for attempt in range(3):
        try:
            llm = get_async_llm()
            chain = _PROMPT | llm.with_structured_output(_NMAOutputRaw)
            raw: _NMAOutputRaw = await chain.ainvoke(signals)
            record_success()

            # ── Step 1: Verify the LLM's exclusion claim ───────────────────────
            #
            # FIX: Check whether the LLM's quoted exclusion_reason actually
            # appears in the stored policy exclusions text before trusting it.
            # This is the primary guard against hallucinated exclusion denials
            # on normal claims.
            exclusions_text: str = signals["_exclusions_text"]
            exclusion_verified = _exclusion_reason_verified(
                raw.exclusion_reason, exclusions_text
            )

            if _parse_bool(raw.exclusion_triggered) and not exclusion_verified:
                logger.warning(
                    "NMA claim=%s | LLM exclusion '%s' NOT verified in policy "
                    "text — treating as no exclusion.",
                    signals["claim_id"], raw.exclusion_reason,
                )

            # ── Step 2: Resolve LLM approval decision ──────────────────────────
            approved_llm = _parse_bool(raw.approved)

            # FIX: No-signal override — when all deterministic signals are
            # clear AND the LLM denies without a verified exclusion, the denial
            # is almost certainly a hallucination. Override to Approved.
            if signals["_all_signals_clear"] and not approved_llm and not exclusion_verified:
                logger.warning(
                    "NMA claim=%s | ALL_SIGNALS_CLEAR=YES but LLM denied with "
                    "no verifiable exclusion — overriding to Approved.",
                    signals["claim_id"],
                )
                approved_llm = True

            # Safety net: if exclusion IS verified but LLM somehow approved,
            # force a denial so exclusion logic is consistent.
            if exclusion_verified and approved_llm:
                logger.warning(
                    "NMA claim=%s | Verified exclusion '%s' but LLM approved — "
                    "overriding to Denied.",
                    signals["claim_id"], raw.exclusion_reason,
                )
                approved_llm = False

            # ── Step 3: Hard deterministic overrides ───────────────────────────
            #
            # FIX: frequent_claims_signal is now a hard override alongside the
            # other three. Previously it was left to the LLM which is
            # non-deterministic. All four signals are deterministic by design
            # and must be enforced unconditionally.
            must_deny = (
                signals["_aggregate_breach_signal"]
                or signals["_collusion_signal"]
                or signals["_staging_signal"]
                or signals["_frequent_claims_signal"]  # FIX: added
            )

            approved_final = approved_llm and not must_deny

            # ── Step 4: Payout calculation ─────────────────────────────────────
            payout_raw = _parse_float(raw.final_payout)
            final_payout = payout_raw if approved_final else 0.0

            # FIX: Payout recalculation guard — llama3:latest sometimes outputs
            # '0.0' for approved claims due to arithmetic confusion in a large
            # prompt. Recalculate deterministically when this happens.
            if approved_final and final_payout <= 0.0:
                remaining = signals["_remaining_limit"]
                deductible = signals["_deductible"]
                loss = signals["_estimated_loss"]
                final_payout = max(0.0, round(min(loss - deductible, remaining), 2))
                logger.info(
                    "NMA claim=%s | LLM payout was 0 despite approval — "
                    "recalculated to %.2f",
                    signals["claim_id"], final_payout,
                )

            # ── Step 5: Fraud score cleanup ────────────────────────────────────
            try:
                fraud_score = max(1, min(10, int(float(raw.fraud_risk_score))))
            except (ValueError, TypeError):
                fraud_score = 5

            if signals["_collusion_signal"] or signals["_staging_signal"]:
                fraud_score = max(fraud_score, 9)
            elif signals["_frequent_claims_signal"]:
                fraud_score = max(fraud_score, 5)
            elif signals["_all_signals_clear"]:
                # FIX: Clamp spurious HIGH fraud scores on clean claims.
                # llama3:latest sometimes escalates risk to 8-9 on a normal
                # rear-end claim with no signals active. Cap at 4 (Medium floor)
                # so downstream reporting is not misleading.
                fraud_score = min(fraud_score, 4)

            logger.info(
                "NMA claim=%s | approved=%s | payout=%.2f | fraud=%d | "
                "signals_clear=%s | exclusion_verified=%s",
                signals["claim_id"], approved_final, final_payout, fraud_score,
                signals["_all_signals_clear"], exclusion_verified,
            )

            return NMAOutput(
                fraud_risk_score=fraud_score,
                fraud_anomalies=raw.fraud_anomalies,
                incident_covered=_parse_bool(raw.incident_covered),
                exclusion_triggered=exclusion_verified,  # trust verification, not LLM flag
                exclusion_reason=raw.exclusion_reason if exclusion_verified else "",
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