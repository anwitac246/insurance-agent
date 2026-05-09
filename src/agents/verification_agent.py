"""
verification_agent.py
---------------------
Document verification agent — async-first, with sync wrapper for compatibility.

Fix 1 (LangChain template error): The old prompt used ChatPromptTemplate with a
JSON example like {"PolicyNumber": "..."} in the system message. LangChain's
template parser treats any {word} as a variable, so "PolicyNumber" was treated
as a missing variable even with double-brace escaping (quotes in the key confuse
the parser). Fixed by building HumanMessage/SystemMessage objects directly,
bypassing the template engine entirely for the system message.

Fix 2 (performance): Single LLM call returning plain JSON — no .with_structured_output().

Fix 3 (429 resilience): If all LLM retries fail we fall back to the raw MongoDB
OCR values rather than crashing, since the data is already structured there.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.tools.groq_client import get_async_llm, record_429, record_success
from src.tools.mongo_client import get_db

logger = logging.getLogger(__name__)

LLM_MAX_RETRIES = 3

# ── System prompt — plain string, never passed through ChatPromptTemplate ──────
# Avoids LangChain treating JSON key names like "PolicyNumber" as {variables}.

_SYSTEM = SystemMessage(content=(
    "You are a document verification specialist for car insurance claims. "
    "Extract and validate the OCR fields exactly as they appear. "
    "Do not infer missing values. "
    "If a field is missing or empty, return an empty string for text fields "
    "and 0.0 for numeric fields.\n\n"
    "Respond ONLY with a raw JSON object. No markdown, no explanation, no preamble. "
    "Required keys and types: PolicyNumber (string), ClaimantName (string), "
    "LossDate (string), RepairShopName (string), TotalEstimate (number)."
))


# ── Schemas ────────────────────────────────────────────────────────────────────

class OcrExtraction(BaseModel):
    PolicyNumber: str = Field(description="Policy number from the document")
    ClaimantName: str = Field(description="Full name of the claimant")
    LossDate: str = Field(description="Date of the loss event")
    RepairShopName: str = Field(description="Name of the repair shop")
    TotalEstimate: float = Field(description="Total repair cost estimate")


class VerificationOutput(BaseModel):
    ocr: OcrExtraction
    name_match: bool
    policy_match: bool
    discrepancies: list[str]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _raw_ocr(ocr_raw: dict) -> OcrExtraction:
    """Build OcrExtraction directly from raw MongoDB values — no LLM needed."""
    return OcrExtraction(
        PolicyNumber=str(ocr_raw.get("PolicyNumber", "")),
        ClaimantName=str(ocr_raw.get("ClaimantName", "")),
        LossDate=str(ocr_raw.get("LossDate", "")),
        RepairShopName=str(ocr_raw.get("RepairShopName", "")),
        TotalEstimate=float(ocr_raw.get("TotalEstimate", 0.0)),
    )


def _parse_ocr_json(text: str, ocr_raw: dict) -> OcrExtraction:
    """Parse LLM plain-text JSON response; fall back to raw OCR on failure."""
    cleaned = re.sub(r"```(?:json)?|```", "", text).strip()
    data: dict = {}
    try:
        data = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group())
            except Exception:
                pass

    if not data:
        logger.warning("verification_agent | JSON parse failed, using raw OCR values")
        return _raw_ocr(ocr_raw)

    try:
        return OcrExtraction(
            PolicyNumber=str(data.get("PolicyNumber", ocr_raw.get("PolicyNumber", ""))),
            ClaimantName=str(data.get("ClaimantName", ocr_raw.get("ClaimantName", ""))),
            LossDate=str(data.get("LossDate", ocr_raw.get("LossDate", ""))),
            RepairShopName=str(data.get("RepairShopName", ocr_raw.get("RepairShopName", ""))),
            TotalEstimate=float(data.get("TotalEstimate", ocr_raw.get("TotalEstimate", 0.0))),
        )
    except (TypeError, ValueError):
        logger.warning("verification_agent | OcrExtraction build failed, using raw OCR")
        return _raw_ocr(ocr_raw)


# ── Async LLM invocation — messages built directly, no ChatPromptTemplate ─────

async def _ainvoke_ocr(ocr_raw: dict) -> OcrExtraction:
    """
    Single LLM call returning plain JSON.
    Uses LangChain message objects directly instead of ChatPromptTemplate to
    avoid the {key} variable-parsing bug with JSON field names.
    Falls back to raw OCR values if all retries fail.
    """
    human = HumanMessage(content=(
        "Parse the following OCR extraction from an insurance claim document "
        "and return a JSON object with the required keys.\n\n"
        f"OCR data:\n{ocr_raw}\n\n"
        "Return the JSON object only — no markdown, no explanation."
    ))
    messages = [_SYSTEM, human]

    last_exc = None
    for attempt in range(LLM_MAX_RETRIES):
        try:
            llm = get_async_llm()
            result = await llm.ainvoke(messages)
            record_success()
            return _parse_ocr_json(result.content, ocr_raw)
        except Exception as exc:
            last_exc = exc
            exc_str = str(exc).lower()
            if "400" in exc_str or "bad request" in exc_str:
                logger.warning(
                    "verification_agent | 400 on attempt %d, using raw OCR: %s",
                    attempt + 1, exc,
                )
                return _raw_ocr(ocr_raw)
            if "429" in exc_str or "rate limit" in exc_str or "rate_limit" in exc_str:
                logger.warning("verification_agent | 429 (attempt %d)", attempt + 1)
                record_429()
                await asyncio.sleep(2 ** attempt)
            else:
                logger.warning(
                    "verification_agent | LLM error (attempt %d): %s", attempt + 1, exc
                )
                await asyncio.sleep(2 ** attempt)

    logger.error(
        "verification_agent | all retries exhausted, using raw OCR: %s", last_exc
    )
    return _raw_ocr(ocr_raw)


# ── Async main ─────────────────────────────────────────────────────────────────

async def arun_verification_agent(state: dict) -> dict:
    t0 = time.perf_counter()
    db = get_db()
    errors: list[str] = list(state.get("errors", []))
    claim_id: str = state.get("claim_id", "")

    claim = db["Active_Claims"].find_one({"claim_id": claim_id})
    if not claim:
        return {
            **state,
            "errors": errors + [f"Claim {claim_id} not found in Active_Claims."],
        }

    raw_data = {k: v for k, v in claim.items() if k != "_id"}
    ocr_raw = claim.get("ocr_extraction", {})

    parsed_ocr: OcrExtraction = await _ainvoke_ocr(ocr_raw)

    customer = db["Customer_Profiles"].find_one({"customer_id": claim["customer_id"]})
    discrepancies: list[str] = []
    name_match = False
    policy_match = False

    if customer:
        stored_name: str = customer.get("full_name", "")
        stored_policy: str = customer.get("policy_id", "")

        name_match = parsed_ocr.ClaimantName.strip().lower() == stored_name.strip().lower()
        policy_match = (
            parsed_ocr.PolicyNumber.strip().upper() == stored_policy[:8].upper()
        )

        if not name_match:
            msg = (
                f"Name mismatch: OCR has '{parsed_ocr.ClaimantName}', "
                f"profile has '{stored_name}'."
            )
            discrepancies.append(msg)
            errors.append(msg)

        if not policy_match:
            msg = (
                f"Policy number mismatch: OCR has '{parsed_ocr.PolicyNumber}', "
                f"profile has '{stored_policy[:8].upper()}'."
            )
            discrepancies.append(msg)
            errors.append(msg)
    else:
        msg = f"No customer profile found for customer_id '{claim['customer_id']}'."
        discrepancies.append(msg)
        errors.append(msg)

    verification_output = VerificationOutput(
        ocr=parsed_ocr,
        name_match=name_match,
        policy_match=policy_match,
        discrepancies=discrepancies,
    )

    sanitized_data = {
        "claim_id": claim_id,
        "customer_id": claim.get("customer_id", ""),
        "policy_id": claim.get("policy_id", ""),
        "incident_type": claim.get("incident_type", ""),
        "narrative": claim.get("narrative", ""),
        "estimated_loss": float(claim.get("estimated_loss", 0)),
        "repair_shop": parsed_ocr.RepairShopName,
        "loss_date": parsed_ocr.LossDate,
        "ocr_total_estimate": parsed_ocr.TotalEstimate,
    }

    elapsed = time.perf_counter() - t0
    logger.info(
        "verification_agent | claim=%s | name_match=%s | policy_match=%s | %.2fs",
        claim_id, name_match, policy_match, elapsed,
    )

    return {
        **state,
        "raw_data": raw_data,
        "sanitized_data": sanitized_data,
        "verification_output": verification_output.model_dump(),
        "customer_profile": {k: v for k, v in (customer or {}).items() if k != "_id"},
        "errors": errors,
    }


# ── Sync wrapper ───────────────────────────────────────────────────────────────

def run_verification_agent(state: dict) -> dict:
    return asyncio.run(arun_verification_agent(state))