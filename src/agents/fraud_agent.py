"""
fraud_agent.py
--------------
Five-signal fraud detection engine — async-first.

Fixes applied:
  - _FraudReportRaw Field descriptions now explicitly forbid True/False/true/false.
  - System prompt includes a CRITICAL format block with correct/wrong examples.
  - _ainvoke_fraud uses a two-step fallback: structured output for safe fields,
    plain-text regex parsing for the three boolean flag fields. This eliminates
    tool_use_failed 400 errors from Groq when the model outputs JSON booleans.
  - All post-LLM deterministic overrides still apply to the final FraudReport.
  - 400 Bad Request errors fail immediately (non-retryable).
"""

import asyncio
import logging
import re
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Literal, Optional

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.tools.groq_client import (
    get_async_llm,
    get_async_fast_llm,
    record_429,
    record_success,
)
from src.tools.mongo_client import get_db

# ── Constants ──────────────────────────────────────────────────────────────────
COLLUSION_SHOP = "Apex AutoBody & Collision"
FREQUENT_CLAIM_THRESHOLD = 3
VELOCITY_WINDOW_DAYS = 365
VELOCITY_THRESHOLD = 3
STAGING_SEVERITY_THRESHOLD = 7
STAGING_ESTIMATE_CAP = 1_000
HIGH_VALUE_LOSS_THRESHOLD = 10_000
LLM_MAX_RETRIES = 3
MAX_HISTORY_RECORDS = 10

logger = logging.getLogger(__name__)


# ── Output schemas ─────────────────────────────────────────────────────────────

class RiskLevel(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


# Internal schema sent to LLM — boolean flags removed entirely to avoid
# tool_use_failed 400 errors. Flags are parsed from plain text instead.
class _FraudReportNoBools(BaseModel):
    """Structured output schema with boolean flags removed to avoid Groq 400s."""
    risk_score: RiskLevel
    anomalies: list[str]
    reasoning: str


# Kept for reference / legacy — not used for LLM output any more.
class _FraudReportRaw(BaseModel):
    risk_score: RiskLevel
    frequent_claims_flag: Literal["yes", "no"] = Field(
        description='Must be the exact string "yes" or the exact string "no". '
                    'NEVER output True, False, true, or false for this field.'
    )
    collusion_flag: Literal["yes", "no"] = Field(
        description='Must be the exact string "yes" or the exact string "no". '
                    'NEVER output True, False, true, or false for this field.'
    )
    staging_flag: Literal["yes", "no"] = Field(
        description='Must be the exact string "yes" or the exact string "no". '
                    'NEVER output True, False, true, or false for this field.'
    )
    anomalies: list[str]
    reasoning: str


# Public schema — proper Python bools, used by downstream agents and metrics
class FraudReport(BaseModel):
    risk_score: RiskLevel
    frequent_claims_flag: bool
    collusion_flag: bool
    staging_flag: bool
    anomalies: list[str]
    reasoning: str


# ── Flag regex parser ──────────────────────────────────────────────────────────

_FLAG_RE = re.compile(
    r'(frequent_claims_flag|collusion_flag|staging_flag)\s*[=:"\s]+\s*(yes|no|true|false)',
    re.IGNORECASE,
)


def _parse_flags(text: str) -> dict[str, bool]:
    """Extract yes/no/true/false flag values from plain LLM text output."""
    flags: dict[str, bool] = {}
    for match in _FLAG_RE.finditer(text):
        key = match.group(1).lower()
        val = match.group(2).lower() in ("yes", "true")
        flags[key] = val
    return flags


# ── Prompts ────────────────────────────────────────────────────────────────────

_SEVERITY_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an accident severity analyst. "
        "Rate the physical severity described in this incident narrative on a scale of "
        "1 (trivial scratch) to 10 (catastrophic multi-vehicle disaster with casualties). "
        "Consider: number of vehicles, emergency services involvement, described damage extent. "
        "Reply with ONLY a single integer between 1 and 10. No explanation, no punctuation.",
    ),
    ("human", "{narrative}"),
])

_FRAUD_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a senior insurance fraud investigator with 20 years of experience. "
        "Synthesize the provided signals and claim context into a final fraud risk assessment. "
        "Reason step by step through every signal before assigning a RiskLevel.\n\n"
        "IMPORTANT: Your anomalies list MUST include every item from the "
        "'Pre-computed Deterministic Anomalies' section, verbatim, plus any additional "
        "ones you identify. You may not drop or omit any pre-computed anomaly.\n\n"
        "CRITICAL — OUTPUT FORMAT RULES:\n"
        "  frequent_claims_flag, collusion_flag, and staging_flag MUST be the exact\n"
        '  string "yes" or the exact string "no".\n'
        "  NEVER output True, False, true, false, or any boolean value for these fields.\n"
        '  Correct:   frequent_claims_flag: "yes"\n'
        '  Correct:   collusion_flag: "no"\n'
        "  WRONG:     staging_flag: False    <- this will cause an API error\n"
        "  WRONG:     staging_flag: true     <- this will cause an API error\n",
    ),
    (
        "human",
        "=== Claim Summary ===\n"
        "Claim ID      : {claim_id}\n"
        "Incident Type : {incident_type}\n"
        "Narrative     : {narrative}\n"
        "Estimated Loss: ${estimated_loss}\n"
        "Repair Shop   : {repair_shop}\n\n"
        "=== Pre-computed Deterministic Anomalies ===\n"
        "{deterministic_anomalies}\n\n"
        "=== Signal Summary ===\n"
        "Frequent Claims Flag (>= {frequent_threshold} denied/flagged): {frequent_claims_flag}\n"
        "Collusion Shop Flag                                           : {collusion_flag}\n"
        "Staging Flag (severity>={severity_threshold}, est<${estimate_cap})  : {staging_flag}\n\n"
        "=== Recent Claim History (last {max_history} records) ===\n"
        "{history_summary}\n\n"
        "=== Customer Profile ===\n"
        "Risk Rating: {risk_rating}\n"
        "NCD Tier   : {ncd_tier}\n\n"
        "Produce a complete FraudReport with risk_score, anomalies, and reasoning. "
        'Also include frequent_claims_flag, collusion_flag, and staging_flag as "yes" or "no".',
    ),
])


# ── Async LLM invocations ──────────────────────────────────────────────────────

async def _ainvoke_severity(inputs: dict) -> int:
    last_exc = None
    for attempt in range(LLM_MAX_RETRIES):
        try:
            llm = get_async_fast_llm()
            chain = _SEVERITY_PROMPT | llm
            result = await chain.ainvoke(inputs)
            text = result.content.strip()
            digits = "".join(ch for ch in text.split()[0] if ch.isdigit())
            score = int(digits) if digits else 5
            record_success()
            return max(1, min(10, score))
        except Exception as exc:
            last_exc = exc
            exc_str = str(exc).lower()
            if "400" in exc_str or "bad request" in exc_str:
                logger.error(
                    "fraud_agent | severity | 400 Bad Request (non-retryable): %s", exc
                )
                raise
            if "429" in exc_str or "rate limit" in exc_str or "rate_limit" in exc_str:
                logger.warning("fraud_agent | severity 429 (attempt %d)", attempt + 1)
                record_429()
                await asyncio.sleep(2 ** attempt)
            else:
                logger.warning(
                    "fraud_agent | severity LLM error (attempt %d): %s", attempt + 1, exc
                )
                await asyncio.sleep(2 ** attempt)
    raise last_exc


async def _ainvoke_fraud(inputs: dict) -> FraudReport:
    """
    Two-step invocation strategy:
      Step 1 — Structured output for safe fields (risk_score, anomalies, reasoning).
               Boolean flags are excluded from the schema to avoid tool_use_failed 400s.
      Step 2 — Plain text call to extract the three flag values via regex.
               Falls back to deterministic values from inputs if parsing fails.

    On 400 errors the function raises immediately (non-retryable).
    On 429 / transient errors it retries with exponential backoff.
    """
    last_exc = None

    for attempt in range(LLM_MAX_RETRIES):
        try:
            llm = get_async_llm()

            # ── Step 1: Structured output (no boolean fields) ──────────────────
            chain_struct = _FRAUD_PROMPT | llm.with_structured_output(_FraudReportNoBools)
            core: _FraudReportNoBools = await chain_struct.ainvoke(inputs)

            # ── Step 2: Plain text to extract flag values ──────────────────────
            try:
                chain_text = _FRAUD_PROMPT | llm
                text_result = await chain_text.ainvoke(inputs)
                flags = _parse_flags(text_result.content)
                logger.debug("fraud_agent | parsed flags from plain text: %s", flags)
            except Exception as flag_exc:
                logger.warning(
                    "fraud_agent | plain-text flag extraction failed, "
                    "falling back to deterministic values: %s", flag_exc
                )
                # Fall back to the deterministic values passed in as inputs
                flags = {
                    "frequent_claims_flag": inputs.get("frequent_claims_flag") == "yes",
                    "collusion_flag": inputs.get("collusion_flag") == "yes",
                    "staging_flag": inputs.get("staging_flag") == "yes",
                }

            record_success()
            return FraudReport(
                risk_score=core.risk_score,
                frequent_claims_flag=flags.get(
                    "frequent_claims_flag",
                    inputs.get("frequent_claims_flag") == "yes",
                ),
                collusion_flag=flags.get(
                    "collusion_flag",
                    inputs.get("collusion_flag") == "yes",
                ),
                staging_flag=flags.get(
                    "staging_flag",
                    inputs.get("staging_flag") == "yes",
                ),
                anomalies=core.anomalies,
                reasoning=core.reasoning,
            )

        except Exception as exc:
            last_exc = exc
            exc_str = str(exc).lower()
            if "400" in exc_str or "bad request" in exc_str:
                logger.error(
                    "fraud_agent | synthesis | 400 Bad Request (non-retryable): %s", exc
                )
                raise
            if "429" in exc_str or "rate limit" in exc_str or "rate_limit" in exc_str:
                logger.warning("fraud_agent | synthesis 429 (attempt %d)", attempt + 1)
                record_429()
                await asyncio.sleep(2 ** attempt)
            else:
                logger.warning(
                    "fraud_agent | synthesis LLM error (attempt %d): %s", attempt + 1, exc
                )
                await asyncio.sleep(2 ** attempt)

    raise last_exc


# ── Helpers ────────────────────────────────────────────────────────────────────

def _summarize_history(records: list[dict], max_records: int = MAX_HISTORY_RECORDS) -> str:
    if not records:
        return "No claim history found."
    recent = sorted(
        records,
        key=lambda r: r.get("incident_date", ""),
        reverse=True,
    )[:max_records]
    lines = [
        f"[{r.get('incident_date', 'N/A')}] {r.get('incident_type', 'Unknown')} | "
        f"Status: {r.get('claim_status', 'Unknown')} | "
        f"Payout: ${r.get('payout_amount', 0):,.2f}"
        for r in recent
    ]
    total = len(records)
    header = (
        f"(Showing {len(recent)} of {total} total records)"
        if total > max_records
        else f"(All {total} records)"
    )
    return header + "\n" + "\n".join(lines)


def _format_anomalies(anomalies: list[str]) -> str:
    if not anomalies:
        return "None detected."
    return "\n".join(f"  - {a}" for a in anomalies)


# ── Async main ─────────────────────────────────────────────────────────────────

async def arun_fraud_agent(state: dict) -> dict:
    t_start = time.perf_counter()
    errors: list[str] = list(state.get("errors", []))
    sanitized: dict = state.get("sanitized_data", {})
    customer_profile: dict = state.get("customer_profile", {})

    claim_id: str = sanitized.get("claim_id", "unknown")
    customer_id: str = sanitized.get("customer_id", "")
    repair_shop: str = sanitized.get("repair_shop", "")
    estimated_loss: float = float(sanitized.get("estimated_loss", 0))
    narrative: str = sanitized.get("narrative", "")

    logger.info(
        "fraud_agent | claim=%s | customer=%s | shop='%s' | loss=%.2f",
        claim_id, customer_id, repair_shop, estimated_loss,
    )

    # ── Fetch claim history ────────────────────────────────────────────────────
    try:
        db = get_db()
        history_records: list[dict] = list(
            db["Claim_History"].find({"customer_id": customer_id}, {"_id": 0})
        )
    except Exception as exc:
        errors.append(f"Fraud agent: Claim_History query failed — {exc}")
        return {**state, "fraud_report": None, "errors": errors}

    # ── Deterministic signals ──────────────────────────────────────────────────
    denied_flagged_count = sum(
        1 for r in history_records
        if r.get("claim_status") in ("Denied", "Fraud_Flagged")
    )
    frequent_claims_flag = denied_flagged_count >= FREQUENT_CLAIM_THRESHOLD
    collusion_flag = COLLUSION_SHOP.lower() in repair_shop.lower()

    cutoff_date = (
        datetime.now() - timedelta(days=VELOCITY_WINDOW_DAYS)
    ).strftime("%Y-%m-%d")
    recent_claim_count = sum(
        1 for r in history_records
        if r.get("incident_date", "") >= cutoff_date
    )

    risk_rating: str = customer_profile.get("risk_rating", "")
    ncd_tier: float = float(customer_profile.get("ncd_tier", 0.0))
    high_risk_high_value = (
        risk_rating == "High Risk" and estimated_loss > HIGH_VALUE_LOSS_THRESHOLD
    )

    # ── Staging signal — plain text severity scoring ───────────────────────────
    staging_flag = False
    severity_score = 0
    try:
        severity_score = await _ainvoke_severity({"narrative": narrative})
        staging_flag = (
            severity_score >= STAGING_SEVERITY_THRESHOLD
            and estimated_loss < STAGING_ESTIMATE_CAP
        )
    except Exception as exc:
        logger.error("claim=%s | Severity scoring failed (non-fatal): %s", claim_id, exc)
        errors.append(f"Fraud agent: staging severity check failed (non-fatal) — {exc}")

    # ── Build deterministic anomalies ──────────────────────────────────────────
    deterministic_anomalies: list[str] = []
    if frequent_claims_flag:
        deterministic_anomalies.append(
            f"Frequent denied/flagged claims: {denied_flagged_count} records "
            f"(threshold: {FREQUENT_CLAIM_THRESHOLD})."
        )
    if collusion_flag:
        deterministic_anomalies.append(
            f"Repair shop matches flagged collusion network: '{repair_shop}'."
        )
    if staging_flag:
        deterministic_anomalies.append(
            f"Staging suspected: narrative severity {severity_score}/10 but "
            f"estimate only ${estimated_loss:,.2f} (threshold: <${STAGING_ESTIMATE_CAP:,})."
        )
    if recent_claim_count >= VELOCITY_THRESHOLD:
        deterministic_anomalies.append(
            f"High claim velocity: {recent_claim_count} claims in the last "
            f"{VELOCITY_WINDOW_DAYS} days (threshold: {VELOCITY_THRESHOLD})."
        )
    if high_risk_high_value:
        deterministic_anomalies.append(
            f"High Risk customer submitting high-value claim of "
            f"${estimated_loss:,.2f} (threshold: ${HIGH_VALUE_LOSS_THRESHOLD:,})."
        )

    # ── LLM synthesis ──────────────────────────────────────────────────────────
    llm_inputs = {
        "claim_id": claim_id,
        "incident_type": sanitized.get("incident_type", ""),
        "narrative": narrative,
        "estimated_loss": f"{estimated_loss:,.2f}",
        "repair_shop": repair_shop,
        "deterministic_anomalies": _format_anomalies(deterministic_anomalies),
        "frequent_threshold": FREQUENT_CLAIM_THRESHOLD,
        "severity_threshold": STAGING_SEVERITY_THRESHOLD,
        "estimate_cap": f"{STAGING_ESTIMATE_CAP:,}",
        "frequent_claims_flag": "yes" if frequent_claims_flag else "no",
        "collusion_flag": "yes" if collusion_flag else "no",
        "staging_flag": "yes" if staging_flag else "no",
        "history_summary": _summarize_history(history_records),
        "max_history": MAX_HISTORY_RECORDS,
        "risk_rating": risk_rating or "Unknown",
        "ncd_tier": ncd_tier,
    }

    try:
        report: FraudReport = await _ainvoke_fraud(llm_inputs)
    except Exception as exc:
        errors.append(f"Fraud agent LLM failed after {LLM_MAX_RETRIES} retries: {exc}")
        return {**state, "fraud_report": None, "errors": errors}

    # ── Merge anomalies ────────────────────────────────────────────────────────
    seen: set[str] = set()
    merged_anomalies: list[str] = []
    for item in deterministic_anomalies + report.anomalies:
        if item not in seen:
            seen.add(item)
            merged_anomalies.append(item)

    # ── Post-LLM risk escalation ───────────────────────────────────────────────
    final_risk = report.risk_score
    escalation_note: Optional[str] = None

    if staging_flag and final_risk != RiskLevel.HIGH:
        final_risk = RiskLevel.HIGH
        escalation_note = (
            f"Risk escalated to HIGH: staging flag overrides LLM assessment "
            f"(severity={severity_score}, estimate=${estimated_loss:,.2f})."
        )
        logger.warning("claim=%s | %s", claim_id, escalation_note)
    elif (frequent_claims_flag or collusion_flag) and final_risk == RiskLevel.LOW:
        final_risk = RiskLevel.MEDIUM
        escalation_note = (
            "Risk escalated to MEDIUM: frequent claims or collusion flag present "
            "but LLM returned LOW."
        )
        logger.warning("claim=%s | %s", claim_id, escalation_note)

    if escalation_note:
        merged_anomalies.insert(0, escalation_note)

    # Always trust the deterministic flags — override whatever the LLM returned
    report = report.model_copy(update={
        "risk_score": final_risk,
        "frequent_claims_flag": frequent_claims_flag,
        "collusion_flag": collusion_flag,
        "staging_flag": staging_flag,
        "anomalies": merged_anomalies,
    })

    if report.risk_score == RiskLevel.HIGH:
        errors.append(
            f"High fraud risk detected for claim {claim_id}. "
            f"Anomalies: {'; '.join(report.anomalies)}"
        )

    elapsed = time.perf_counter() - t_start
    logger.info(
        "fraud_agent | claim=%s | done in %.2fs | risk=%s | anomalies=%d",
        claim_id, elapsed, report.risk_score, len(report.anomalies),
    )

    return {**state, "fraud_report": report.model_dump(), "errors": errors}


# ── Sync wrapper ───────────────────────────────────────────────────────────────

def run_fraud_agent(state: dict) -> dict:
    return asyncio.run(arun_fraud_agent(state))