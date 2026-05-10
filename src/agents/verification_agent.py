"""
verification_agent.py
---------------------
Document verification agent — async-first, with sync wrapper for compatibility.

v2 changes
----------
- REMOVED the LLM call for OCR parsing entirely. The OCR data stored in
  Active_Claims.ocr_extraction is already a structured dict — there is
  no unstructured text to "parse". Calling an LLM to echo back the same
  key/value pairs wasted 1 LLM call per claim and was the single largest
  source of unnecessary latency and rate-limit pressure.
  The raw MongoDB values are now consumed directly via _raw_ocr().

- LLM call count per claim: 1  →  0  (verification is now fully deterministic)
"""

from __future__ import annotations

import asyncio
import logging
import time

from pydantic import BaseModel, Field

from src.tools.mongo_client import get_db

logger = logging.getLogger(__name__)


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
    """
    Build OcrExtraction directly from the structured MongoDB values.
    No LLM needed — this is already validated JSON, not raw OCR text.
    """
    return OcrExtraction(
        PolicyNumber=str(ocr_raw.get("PolicyNumber", "")),
        ClaimantName=str(ocr_raw.get("ClaimantName", "")),
        LossDate=str(ocr_raw.get("LossDate", "")),
        RepairShopName=str(ocr_raw.get("RepairShopName", "")),
        TotalEstimate=float(ocr_raw.get("TotalEstimate", 0.0)),
    )


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
    ocr_raw: dict = claim.get("ocr_extraction", {})

    # ── Deterministic OCR extraction — zero LLM calls ─────────────────────────
    parsed_ocr: OcrExtraction = _raw_ocr(ocr_raw)

    # ── Cross-validate against Customer_Profiles ──────────────────────────────
    customer = db["Customer_Profiles"].find_one({"customer_id": claim["customer_id"]})
    discrepancies: list[str] = []
    name_match = False
    policy_match = False

    if customer:
        stored_name: str = customer.get("full_name", "")
        stored_policy: str = customer.get("policy_id", "")

        name_match = (
            parsed_ocr.ClaimantName.strip().lower() == stored_name.strip().lower()
        )
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
        msg = (
            f"No customer profile found for customer_id '{claim['customer_id']}'."
        )
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
        "verification_agent | claim=%s | name_match=%s | policy_match=%s | %.2fs (0 LLM calls)",
        claim_id, name_match, policy_match, elapsed,
    )

    return {
        **state,
        "raw_data": raw_data,
        "sanitized_data": sanitized_data,
        "verification_output": verification_output.model_dump(),
        "customer_profile": {
            k: v for k, v in (customer or {}).items() if k != "_id"
        },
        "errors": errors,
    }


# ── Sync wrapper ───────────────────────────────────────────────────────────────

def run_verification_agent(state: dict) -> dict:
    return asyncio.run(arun_verification_agent(state))