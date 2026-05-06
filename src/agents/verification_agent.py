"""
verification_agent.py
---------------------
Document verification agent — async-first, with sync wrapper for compatibility.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.tools.groq_client import get_async_llm, record_429, record_success
from src.tools.mongo_client import get_db

logger = logging.getLogger(__name__)

LLM_MAX_RETRIES = 3


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


# ── Prompt ─────────────────────────────────────────────────────────────────────

_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a document verification specialist for car insurance claims. "
        "Extract and validate the OCR fields exactly as they appear. "
        "Do not infer missing values. "
        "If a field is missing or empty, return an empty string for text fields "
        "and 0.0 for numeric fields.",
    ),
    (
        "human",
        "Parse and validate the following OCR extraction from an insurance claim document:\n\n"
        "{ocr_json}\n\n"
        "Return the structured fields.",
    ),
])


# ── Async LLM invocation ───────────────────────────────────────────────────────

async def _ainvoke_ocr(inputs: dict) -> OcrExtraction:
    last_exc = None
    for attempt in range(LLM_MAX_RETRIES):
        try:
            llm = get_async_llm()
            chain = _prompt | llm.with_structured_output(OcrExtraction)
            result = await chain.ainvoke(inputs)
            record_success()
            return result
        except Exception as exc:
            last_exc = exc
            exc_str = str(exc).lower()
            if "429" in exc_str or "rate limit" in exc_str or "rate_limit" in exc_str:
                logger.warning("verification_agent | 429 detected (attempt %d)", attempt + 1)
                record_429()
                await asyncio.sleep(2 ** attempt)
            else:
                logger.warning("verification_agent | LLM error (attempt %d): %s", attempt + 1, exc)
                await asyncio.sleep(2 ** attempt)
    raise last_exc


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

    try:
        parsed_ocr: OcrExtraction = await _ainvoke_ocr({"ocr_json": str(ocr_raw)})
    except Exception as exc:
        logger.exception("OCR parsing failed for claim %s", claim_id)
        return {
            **state,
            "raw_data": raw_data,
            "errors": errors + [f"OCR parsing failed: {exc}"],
        }

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
        "errors": errors,
    }


# ── Sync wrapper (for LangGraph compatibility) ─────────────────────────────────

def run_verification_agent(state: dict) -> dict:
    return asyncio.run(arun_verification_agent(state))