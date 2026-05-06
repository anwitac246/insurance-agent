"""
fraud_agent.py
--------------
Five-signal fraud detection engine for car insurance claims.

Signals computed deterministically before any LLM call:
  1. Frequent denied/flagged claims  (MongoDB Claim_History)
  2. Collusion shop match            (repair shop name)
  3. Staging detection               (narrative severity vs. estimate — lightweight LLM)
  4. Claim velocity                  (claims in last 365 days)
  5. Risk/loss profile mismatch      (High Risk customer + high-value claim)

A single synthesis LLM call combines all signals into a structured FraudReport.
Post-LLM escalation rules ensure deterministic signals cannot be downgraded by
the LLM.  All pre-computed anomalies are merged back into the final report so
they cannot be silently dropped.
"""

import logging
import time
from datetime import datetime, timedelta
from enum import Enum
from functools import lru_cache
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.tools.mongo_client import get_db

# ── Constants ──────────────────────────────────────────────────────────────────
GROQ_MODEL = "llama-3.3-70b-versatile"
COLLUSION_SHOP = "Apex AutoBody & Collision"

FREQUENT_CLAIM_THRESHOLD = 3       # denied/flagged claims to trigger flag
VELOCITY_WINDOW_DAYS = 365         # lookback window for claim velocity
VELOCITY_THRESHOLD = 3             # claims in window to trigger anomaly
STAGING_SEVERITY_THRESHOLD = 7    # narrative severity score (1–10) to suspect staging
STAGING_ESTIMATE_CAP = 1_000      # max estimate (USD) paired with high severity
HIGH_VALUE_LOSS_THRESHOLD = 10_000 # loss above which High Risk profile is flagged

LLM_MAX_RETRIES = 3
LLM_RETRY_WAIT_MIN = 2
LLM_RETRY_WAIT_MAX = 10

# ── Logging ────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)


# ── Lazy LLM singleton ─────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _get_llm() -> ChatGroq:
    return ChatGroq(model=GROQ_MODEL, temperature=0, request_timeout=30)


# ── Output schemas ─────────────────────────────────────────────────────────────
class RiskLevel(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


class FraudReport(BaseModel):
    risk_score: RiskLevel
    frequent_claims_flag: bool
    collusion_flag: bool
    staging_flag: bool
    anomalies: list[str]
    reasoning: str


class _SeverityScore(BaseModel):
    severity: int   # 1 (minor) – 10 (catastrophic)


# ── Prompts ────────────────────────────────────────────────────────────────────
_SEVERITY_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an accident severity analyst. "
        "Rate the physical severity described in this incident narrative on a scale of "
        "1 (trivial scratch) to 10 (catastrophic multi-vehicle disaster with casualties). "
        "Consider: number of vehicles, emergency services involvement, described damage extent. "
        "Return only the integer score — no explanation.",
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
        "ones you identify. You may not drop or omit any pre-computed anomaly.",
    ),
    (
        "human",
        "=== Claim Summary ===\n"
        "Claim ID      : {claim_id}\n"
        "Incident Type : {incident_type}\n"
        "Narrative     : {narrative}\n"
        "Estimated Loss: ${estimated_loss:,.2f}\n"
        "Repair Shop   : {repair_shop}\n\n"
        "=== Pre-computed Deterministic Anomalies ===\n"
        "{deterministic_anomalies}\n\n"
        "=== Signal Summary ===\n"
        "Frequent Claims Flag (≥{frequent_threshold} denied/flagged): {frequent_claims_flag}\n"
        "Collusion Shop Flag                                        : {collusion_flag}\n"
        "Staging Flag (severity≥{severity_threshold}, est<${estimate_cap})  : {staging_flag}\n\n"
        "=== Full Claim History ===\n"
        "{history_summary}\n\n"
        "=== Customer Profile ===\n"
        "Risk Rating: {risk_rating}\n"
        "NCD Tier   : {ncd_tier}\n\n"
        "Produce a complete FraudReport.",
    ),
])


# ── Retry-wrapped LLM calls ────────────────────────────────────────────────────
@retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(LLM_MAX_RETRIES),
    wait=wait_exponential(min=LLM_RETRY_WAIT_MIN, max=LLM_RETRY_WAIT_MAX),
    reraise=True,
)
def _invoke_severity(chain, inputs: dict) -> _SeverityScore:
    return chain.invoke(inputs)


@retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(LLM_MAX_RETRIES),
    wait=wait_exponential(min=LLM_RETRY_WAIT_MIN, max=LLM_RETRY_WAIT_MAX),
    reraise=True,
)
def _invoke_fraud(chain, inputs: dict) -> FraudReport:
    return chain.invoke(inputs)


# ── Helpers ────────────────────────────────────────────────────────────────────
def _summarize_history(records: list[dict]) -> str:
    if not records:
        return "No claim history found."
    lines = [
        f"[{r.get('incident_date', 'N/A')}] {r.get('incident_type', 'Unknown')} | "
        f"Status: {r.get('claim_status', 'Unknown')} | "
        f"Payout: ${r.get('payout_amount', 0):,.2f}"
        for r in records
    ]
    return "\n".join(lines)


def _format_anomalies(anomalies: list[str]) -> str:
    if not anomalies:
        return "None detected."
    return "\n".join(f"  • {a}" for a in anomalies)


# ── Main agent function ────────────────────────────────────────────────────────
def run_fraud_agent(state: dict) -> dict:
    """
    Populates state["fraud_report"] with a FraudReport dict.
    Appends to state["errors"] on HIGH risk detection.
    Never raises — always returns a valid state dict.
    """
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
        logger.debug("claim=%s | fetched %d history records", claim_id, len(history_records))
    except Exception as exc:
        errors.append(f"Fraud agent: Claim_History query failed — {exc}")
        logger.exception("MongoDB query failed for claim %s", claim_id)
        return {**state, "fraud_report": None, "errors": errors}

    # ── Signal 1: Frequent denied/flagged claims ───────────────────────────────
    denied_flagged_count = sum(
        1 for r in history_records
        if r.get("claim_status") in ("Denied", "Fraud_Flagged")
    )
    frequent_claims_flag = denied_flagged_count >= FREQUENT_CLAIM_THRESHOLD
    logger.debug("claim=%s | denied_flagged=%d | frequent=%s", claim_id, denied_flagged_count, frequent_claims_flag)

    # ── Signal 2: Collusion shop ───────────────────────────────────────────────
    collusion_flag = COLLUSION_SHOP.lower() in repair_shop.lower()
    if collusion_flag:
        logger.warning("claim=%s | collusion shop detected: '%s'", claim_id, repair_shop)

    # ── Signal 3: Staging detection ───────────────────────────────────────────
    staging_flag = False
    severity_score = 0
    severity_chain = _SEVERITY_PROMPT | _get_llm().with_structured_output(_SeverityScore)
    try:
        sev_result = _invoke_severity(severity_chain, {"narrative": narrative})
        severity_score = max(1, min(10, sev_result.severity))  # clamp to [1, 10]
        staging_flag = severity_score >= STAGING_SEVERITY_THRESHOLD and estimated_loss < STAGING_ESTIMATE_CAP
        logger.debug(
            "claim=%s | severity=%d | estimate=%.2f | staging=%s",
            claim_id, severity_score, estimated_loss, staging_flag,
        )
    except Exception as exc:
        # Non-fatal: log and continue without the staging signal
        logger.error("claim=%s | Severity scoring failed (staging signal skipped): %s", claim_id, exc)
        errors.append(f"Fraud agent: staging severity check failed (non-fatal) — {exc}")

    # ── Signal 4: Claim velocity ───────────────────────────────────────────────
    cutoff_date = (datetime.now() - timedelta(days=VELOCITY_WINDOW_DAYS)).strftime("%Y-%m-%d")
    recent_claim_count = sum(
        1 for r in history_records
        if r.get("incident_date", "") >= cutoff_date
    )

    # ── Signal 5: Risk/loss profile mismatch ──────────────────────────────────
    risk_rating: str = customer_profile.get("risk_rating", "")
    ncd_tier: float = float(customer_profile.get("ncd_tier", 0.0))
    high_risk_high_value = risk_rating == "High Risk" and estimated_loss > HIGH_VALUE_LOSS_THRESHOLD

    # ── Build deterministic anomalies (ground-truth — LLM cannot remove these) ─
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

    logger.info(
        "claim=%s | signals: frequent=%s collusion=%s staging=%s velocity=%d high_risk_hv=%s | anomalies=%d",
        claim_id, frequent_claims_flag, collusion_flag, staging_flag,
        recent_claim_count, high_risk_high_value, len(deterministic_anomalies),
    )

    # ── Synthesis LLM call ─────────────────────────────────────────────────────
    fraud_chain = _FRAUD_PROMPT | _get_llm().with_structured_output(FraudReport)
    llm_inputs = {
        "claim_id": claim_id,
        "incident_type": sanitized.get("incident_type", ""),
        "narrative": narrative,
        "estimated_loss": estimated_loss,
        "repair_shop": repair_shop,
        "deterministic_anomalies": _format_anomalies(deterministic_anomalies),
        "frequent_threshold": FREQUENT_CLAIM_THRESHOLD,
        "severity_threshold": STAGING_SEVERITY_THRESHOLD,
        "estimate_cap": STAGING_ESTIMATE_CAP,
        "frequent_claims_flag": frequent_claims_flag,
        "collusion_flag": collusion_flag,
        "staging_flag": staging_flag,
        "history_summary": _summarize_history(history_records),
        "risk_rating": risk_rating or "Unknown",
        "ncd_tier": ncd_tier,
    }

    try:
        report: FraudReport = _invoke_fraud(fraud_chain, llm_inputs)
        logger.info("claim=%s | LLM risk_score=%s", claim_id, report.risk_score)
    except Exception as exc:
        errors.append(f"Fraud agent LLM failed after {LLM_MAX_RETRIES} retries: {exc}")
        logger.exception("Fraud synthesis LLM failed for claim %s", claim_id)
        return {**state, "fraud_report": None, "errors": errors}

    # ── Merge anomalies: pre-computed signals cannot be dropped by the LLM ────
    # Dedup while preserving order: deterministic signals listed first
    seen: set[str] = set()
    merged_anomalies: list[str] = []
    for item in deterministic_anomalies + report.anomalies:
        if item not in seen:
            seen.add(item)
            merged_anomalies.append(item)

    # ── Post-LLM risk escalation (deterministic overrides) ────────────────────
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