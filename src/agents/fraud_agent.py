"""
fraud_agent.py
--------------
Five-signal fraud detection engine — async-first.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.tools.groq_client import (
    get_async_llm,
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


def _parse_bool(val) -> bool:
    return str(val).lower().strip() in ("yes", "true", "1")


class _FraudReportRaw(BaseModel):
    """
    Raw LLM output schema.

    narrative_severity_score: int 1-10 — replaces the old separate severity call.
    Boolean flags typed as str — Pydantic v2 coerces True/False/"yes"/"no"
    without raising, preventing Groq 400 Bad Request tool_use_failed errors.
    """
    narrative_severity_score: int = Field(
        ge=1, le=10,
        description=(
            "Physical severity of the incident on a scale of 1 (trivial) "
            "to 10 (catastrophic multi-vehicle disaster). "
            "Consider: number of vehicles, emergency services, described damage."
        ),
    )
    risk_score: RiskLevel
    frequent_claims_flag: str = Field(
        description='Output "yes" if the flag is active, "no" otherwise.'
    )
    collusion_flag: str = Field(
        description='Output "yes" if the flag is active, "no" otherwise.'
    )
    staging_flag: str = Field(
        description='Output "yes" if the flag is active, "no" otherwise.'
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


# ── Single combined prompt ─────────────────────────────────────────────────────
# Severity scoring is now a sub-task inside this prompt.
# The LLM emits `narrative_severity_score` alongside all fraud fields.

_FRAUD_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a senior insurance fraud investigator with 20 years of experience.\n\n"
        "Your tasks in a SINGLE pass:\n"
        "  1. Rate the physical severity of the incident narrative on a scale of 1–10:\n"
        "       1 = trivial scratch, 10 = catastrophic multi-vehicle disaster.\n"
        "       Consider: number of vehicles, emergency services, described damage.\n"
        "       Emit this as `narrative_severity_score` (integer).\n"
        "  2. Synthesize ALL provided signals into a final fraud risk assessment.\n\n"
        "Rules:\n"
        "  - Your anomalies list MUST include every item from "
        "'Pre-computed Deterministic Anomalies' verbatim.\n"
        "  - For frequent_claims_flag, collusion_flag, and staging_flag output "
        "the string 'yes' or 'no' based on the pre-computed signals provided.\n"
        "  - Reason step by step through every signal before assigning RiskLevel.",
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
        "Staging Check (severity>={severity_threshold}, est<${estimate_cap})\n"
        "  → You must score severity first, then set staging_flag accordingly.\n\n"
        "=== Recent Claim History (last {max_history} records) ===\n"
        "{history_summary}\n\n"
        "=== Customer Profile ===\n"
        "Risk Rating: {risk_rating}\n"
        "NCD Tier   : {ncd_tier}\n\n"
        "First score narrative_severity_score (1-10), then produce a complete "
        "FraudReport with risk_score, anomalies, reasoning, and all three flag fields.",
    ),
])


# ── Async LLM invocation — single call ────────────────────────────────────────

async def _ainvoke_fraud(inputs: dict) -> tuple[int, FraudReport]:
    """
    Single structured-output call.
    Returns (severity_score, FraudReport).
    Boolean flags coerced from str by Pydantic v2.
    """
    last_exc: Optional[Exception] = None

    for attempt in range(LLM_MAX_RETRIES):
        try:
            llm = get_async_llm()
            chain = _FRAUD_PROMPT | llm.with_structured_output(_FraudReportRaw)
            raw: _FraudReportRaw = await chain.ainvoke(inputs)
            record_success()

            severity = max(1, min(10, int(raw.narrative_severity_score)))
            report = FraudReport(
                risk_score=raw.risk_score,
                frequent_claims_flag=_parse_bool(raw.frequent_claims_flag),
                collusion_flag=_parse_bool(raw.collusion_flag),
                staging_flag=_parse_bool(raw.staging_flag),
                anomalies=raw.anomalies,
                reasoning=raw.reasoning,
            )
            return severity, report

        except Exception as exc:
            last_exc = exc
            exc_str = str(exc).lower()
            if "400" in exc_str or "bad request" in exc_str:
                logger.error(
                    "fraud_agent | 400 Bad Request (non-retryable): %s", exc
                )
                raise
            if "429" in exc_str or "rate limit" in exc_str or "rate_limit" in exc_str:
                logger.warning("fraud_agent | 429 (attempt %d)", attempt + 1)
                record_429()
                await asyncio.sleep(2 ** attempt)
            else:
                logger.warning(
                    "fraud_agent | LLM error (attempt %d): %s", attempt + 1, exc
                )
                await asyncio.sleep(2 ** attempt)

    raise last_exc  # type: ignore[misc]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _summarize_history(
    records: list[dict], max_records: int = MAX_HISTORY_RECORDS
) -> str:
    if not records:
        return "No claim history found."
    recent = sorted(
        records, key=lambda r: r.get("incident_date", ""), reverse=True
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

    # ── Fetch claim history from MongoDB ──────────────────────────────────────
    try:
        db = get_db()
        history_records: list[dict] = list(
            db["Claim_History"].find({"customer_id": customer_id}, {"_id": 0})
        )
    except Exception as exc:
        errors.append(f"Fraud agent: Claim_History query failed — {exc}")
        return {**state, "fraud_report": None, "errors": errors}

    # ── Deterministic pre-compute signals ─────────────────────────────────────
    denied_flagged_count = sum(
        1 for r in history_records
        if r.get("claim_status") in ("Denied", "Fraud_Flagged")
    )
    frequent_claims_flag_det = denied_flagged_count >= FREQUENT_CLAIM_THRESHOLD
    collusion_flag_det = COLLUSION_SHOP.lower() in repair_shop.lower()

    cutoff_date = (
        datetime.now() - timedelta(days=VELOCITY_WINDOW_DAYS)
    ).strftime("%Y-%m-%d")
    recent_claim_count = sum(
        1 for r in history_records if r.get("incident_date", "") >= cutoff_date
    )

    risk_rating: str = customer_profile.get("risk_rating", "")
    ncd_tier: float = float(customer_profile.get("ncd_tier", 0.0))
    high_risk_high_value = (
        risk_rating == "High Risk" and estimated_loss > HIGH_VALUE_LOSS_THRESHOLD
    )

    # Build deterministic anomalies list BEFORE the LLM call.
    # Staging anomaly is added after we get severity back from the LLM.
    deterministic_anomalies: list[str] = []
    if frequent_claims_flag_det:
        deterministic_anomalies.append(
            f"Frequent denied/flagged claims: {denied_flagged_count} records "
            f"(threshold: {FREQUENT_CLAIM_THRESHOLD})."
        )
    if collusion_flag_det:
        deterministic_anomalies.append(
            f"Repair shop matches flagged collusion network: '{repair_shop}'."
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

    # ── Single LLM call — severity scoring + fraud synthesis combined ─────────
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
        "frequent_claims_flag": "yes" if frequent_claims_flag_det else "no",
        "collusion_flag": "yes" if collusion_flag_det else "no",
        "history_summary": _summarize_history(history_records),
        "max_history": MAX_HISTORY_RECORDS,
        "risk_rating": risk_rating or "Unknown",
        "ncd_tier": ncd_tier,
    }

    try:
        severity_score, report = await _ainvoke_fraud(llm_inputs)
    except Exception as exc:
        errors.append(
            f"Fraud agent LLM failed after {LLM_MAX_RETRIES} retries: {exc}"
        )
        return {**state, "fraud_report": None, "errors": errors}

    # ── Compute staging flag NOW that we have severity from the LLM ───────────
    staging_flag_det = (
        severity_score >= STAGING_SEVERITY_THRESHOLD
        and estimated_loss < STAGING_ESTIMATE_CAP
    )
    if staging_flag_det:
        staging_anomaly = (
            f"Staging suspected: narrative severity {severity_score}/10 but "
            f"estimate only ${estimated_loss:,.2f} (threshold: <${STAGING_ESTIMATE_CAP:,})."
        )
        deterministic_anomalies.append(staging_anomaly)

    # ── Merge anomalies (deterministic first, then LLM additions) ────────────
    seen: set[str] = set()
    merged_anomalies: list[str] = []
    for item in deterministic_anomalies + report.anomalies:
        if item not in seen:
            seen.add(item)
            merged_anomalies.append(item)

    # ── Post-LLM risk escalation (deterministic overrides) ───────────────────
    final_risk = report.risk_score
    escalation_note: Optional[str] = None

    if staging_flag_det and final_risk != RiskLevel.HIGH:
        final_risk = RiskLevel.HIGH
        escalation_note = (
            f"Risk escalated to HIGH: staging flag overrides LLM assessment "
            f"(severity={severity_score}, estimate=${estimated_loss:,.2f})."
        )
        logger.warning("claim=%s | %s", claim_id, escalation_note)
    elif (
        (frequent_claims_flag_det or collusion_flag_det)
        and final_risk == RiskLevel.LOW
    ):
        final_risk = RiskLevel.MEDIUM
        escalation_note = (
            "Risk escalated to MEDIUM: frequent claims or collusion flag present "
            "but LLM returned LOW."
        )
        logger.warning("claim=%s | %s", claim_id, escalation_note)

    if escalation_note:
        merged_anomalies.insert(0, escalation_note)

    # Always enforce deterministic flag values — override whatever the LLM said
    report = report.model_copy(update={
        "risk_score": final_risk,
        "frequent_claims_flag": frequent_claims_flag_det,
        "collusion_flag": collusion_flag_det,
        "staging_flag": staging_flag_det,
        "anomalies": merged_anomalies,
    })

    if report.risk_score == RiskLevel.HIGH:
        errors.append(
            f"High fraud risk detected for claim {claim_id}. "
            f"Anomalies: {'; '.join(report.anomalies)}"
        )

    elapsed = time.perf_counter() - t_start
    logger.info(
        "fraud_agent | claim=%s | done in %.2fs (1 LLM call) | "
        "severity=%d | risk=%s | anomalies=%d",
        claim_id, elapsed, severity_score, report.risk_score, len(report.anomalies),
    )

    return {**state, "fraud_report": report.model_dump(), "errors": errors}


# ── Sync wrapper ───────────────────────────────────────────────────────────────

def run_fraud_agent(state: dict) -> dict:
    return asyncio.run(arun_fraud_agent(state))