import os
from datetime import datetime, timedelta
from enum import Enum
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from src.tools.mongo_client import get_db

GROQ_MODEL = "llama-3.3-70b-versatile"
COLLUSION_SHOP = "Apex AutoBody & Collision"
FREQUENT_CLAIM_THRESHOLD = 3
VELOCITY_WINDOW_DAYS = 365
VELOCITY_THRESHOLD = 3
STAGING_SEVERITY_THRESHOLD = 7
STAGING_ESTIMATE_CAP = 1000
HIGH_VALUE_LOSS_THRESHOLD = 10000

_llm = ChatGroq(model=GROQ_MODEL, temperature=0)


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
    severity: int


_severity_prompt = ChatPromptTemplate.from_messages([
    ("system", "Rate the physical severity of this incident narrative on a scale of 1 (minor) to 10 (catastrophic). Return only the integer score."),
    ("human", "{narrative}"),
])
_severity_chain = _severity_prompt | _llm.with_structured_output(_SeverityScore)

_fraud_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a senior insurance fraud investigator. Synthesize all provided signals into a final fraud risk assessment. "
        "Reason step by step through each signal before assigning a RiskLevel (Low/Medium/High).",
    ),
    (
        "human",
        "=== Claim Summary ===\n"
        "Claim ID: {claim_id}\n"
        "Incident Type: {incident_type}\n"
        "Narrative: {narrative}\n"
        "Estimated Loss: ${estimated_loss}\n"
        "Repair Shop: {repair_shop}\n\n"
        "=== Pre-computed Signals ===\n"
        "Frequent Claims Flag: {frequent_claims_flag}\n"
        "Collusion Shop Flag: {collusion_flag}\n"
        "Staging Flag: {staging_flag}\n"
        "Anomalies Detected: {anomalies}\n\n"
        "=== Claim History Summary ===\n"
        "{history_summary}\n\n"
        "=== Customer Profile ===\n"
        "Risk Rating: {risk_rating}\n"
        "NCD Tier: {ncd_tier}\n\n"
        "Produce a complete FraudReport.",
    ),
])
_fraud_chain = _fraud_prompt | _llm.with_structured_output(FraudReport)


def _summarize_history(records: list[dict]) -> str:
    if not records:
        return "No claim history found."
    lines = []
    for r in records:
        lines.append(
            f"[{r.get('incident_date', 'N/A')}] {r.get('incident_type', 'Unknown')} — "
            f"Status: {r.get('claim_status', 'Unknown')} — Payout: ${r.get('payout_amount', 0):,.2f}"
        )
    return "\n".join(lines)


def run_fraud_agent(state: dict) -> dict:
    errors: list[str] = list(state.get("errors", []))
    sanitized = state.get("sanitized_data", {})
    customer_profile: dict = state.get("customer_profile", {})

    customer_id: str = sanitized.get("customer_id", "")
    repair_shop: str = sanitized.get("repair_shop", "")
    estimated_loss: float = float(sanitized.get("estimated_loss", 0))
    narrative: str = sanitized.get("narrative", "")

    db = get_db()
    history_records: list[dict] = list(db["Claim_History"].find({"customer_id": customer_id}))

    # Signal 1 — Frequent Claims
    denied_flagged_count = sum(
        1 for r in history_records if r.get("claim_status") in ("Denied", "Fraud_Flagged")
    )
    frequent_claims_flag = denied_flagged_count >= FREQUENT_CLAIM_THRESHOLD

    # Signal 2 — Collusion Shop
    collusion_flag = COLLUSION_SHOP.lower() in repair_shop.lower()

    # Signal 3 — Staging Detection
    staging_flag = False
    try:
        severity_result = _severity_chain.invoke({"narrative": narrative})
        severity_score = severity_result.severity
        if severity_score >= STAGING_SEVERITY_THRESHOLD and estimated_loss < STAGING_ESTIMATE_CAP:
            staging_flag = True
    except Exception:
        severity_score = 0

    # Signal 4 — Claim Velocity
    cutoff = (datetime.now() - timedelta(days=VELOCITY_WINDOW_DAYS)).strftime("%Y-%m-%d")
    recent_claims = [r for r in history_records if r.get("incident_date", "") >= cutoff]
    anomalies: list[str] = []
    if len(recent_claims) >= VELOCITY_THRESHOLD:
        anomalies.append(f"High claim velocity: {len(recent_claims)} claims in 12 months.")

    # Signal 5 — NCD/Risk Profile Mismatch
    risk_rating: str = customer_profile.get("risk_rating", "")
    ncd_tier: float = float(customer_profile.get("ncd_tier", 0))
    if risk_rating == "High Risk" and estimated_loss > HIGH_VALUE_LOSS_THRESHOLD:
        anomalies.append(
            f"High Risk customer submitting high-value claim of ${estimated_loss:,.2f}."
        )

    if frequent_claims_flag:
        anomalies.append(f"Frequent denied/flagged claims: {denied_flagged_count} records.")
    if collusion_flag:
        anomalies.append(f"Claim directed to flagged collusion shop: {repair_shop}.")
    if staging_flag:
        anomalies.append(
            f"Staging suspected: narrative severity {severity_score}/10 but estimate only ${estimated_loss:,.2f}."
        )

    try:
        report: FraudReport = _fraud_chain.invoke({
            "claim_id": sanitized.get("claim_id", ""),
            "incident_type": sanitized.get("incident_type", ""),
            "narrative": narrative,
            "estimated_loss": estimated_loss,
            "repair_shop": repair_shop,
            "frequent_claims_flag": frequent_claims_flag,
            "collusion_flag": collusion_flag,
            "staging_flag": staging_flag,
            "anomalies": anomalies if anomalies else ["None"],
            "history_summary": _summarize_history(history_records),
            "risk_rating": risk_rating,
            "ncd_tier": ncd_tier,
        })
    except Exception as exc:
        errors.append(f"Fraud agent LLM failed: {exc}")
        return {**state, "fraud_report": None, "errors": errors}

    # Risk escalation overrides
    if staging_flag and report.risk_score != RiskLevel.HIGH:
        report = report.model_copy(update={"risk_score": RiskLevel.HIGH})
    elif (frequent_claims_flag or collusion_flag) and report.risk_score == RiskLevel.LOW:
        report = report.model_copy(update={"risk_score": RiskLevel.MEDIUM})

    if report.risk_score == RiskLevel.HIGH:
        errors.append(
            f"High fraud risk detected. Anomalies: {'; '.join(report.anomalies)}"
        )

    return {**state, "fraud_report": report.model_dump(), "errors": errors}