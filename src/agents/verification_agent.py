"""
verification_agent.py
---------------------
Document verification agent — async-first, with sync wrapper for compatibility.

BUG FIXES vs previous version
------------------------------
1. Name mismatch and policy-number mismatch are now WARNINGS, not ERRORS.
   They are stored in state["warnings"] and still surfaced to the decision
   agent as context, but they no longer trigger should_continue() to route
   the claim to failure_node.  Previously these minor OCR discrepancies
   caused normal claims to be hard-denied at the verification stage,
   which was the primary driver of false denials in the MAS.

2. Only truly fatal conditions — claim not found in Active_Claims, or
   customer profile not found — are written to state["errors"], which
   will trigger failure_node routing.

3. The `warnings` list is added to the returned state so the graph's
   new should_continue() and decision_agent can both see it.
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
    warnings: list[str] = list(state.get("warnings", []))
    claim_id: str = state.get("claim_id", "")

    claim = db["Active_Claims"].find_one({"claim_id": claim_id})
    if not claim:
        # FATAL — claim does not exist at all
        return {
            **state,
            "warnings": warnings,
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
            # BUG FIX: was errors.append(msg) → now a WARNING, not a fatal error.
            # A name mismatch is an OCR quality issue, not a reason to deny the
            # claim outright before policy and fraud agents have evaluated it.
            warnings.append(msg)
            logger.warning("claim=%s | non-fatal discrepancy: %s", claim_id, msg)

        if not policy_match:
            msg = (
                f"Policy number mismatch: OCR has '{parsed_ocr.PolicyNumber}', "
                f"profile has '{stored_policy[:8].upper()}'."
            )
            discrepancies.append(msg)
            # BUG FIX: same — demote to warning.
            warnings.append(msg)
            logger.warning("claim=%s | non-fatal discrepancy: %s", claim_id, msg)
    else:
        # FATAL — no customer profile means we cannot look up the policy
        msg = (
            f"No customer profile found for customer_id '{claim['customer_id']}'."
        )
        discrepancies.append(msg)
        errors.append(msg)   # this IS fatal

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
        # BUG FIX (used by fraud_agent for staging detection):
        # ocr_total_estimate is the figure extracted from the repair document.
        # For staged accidents seed_data sets this to $200 while estimated_loss
        # is $8k–$20k.  The fraud agent must compare these two values, so both
        # must be present in sanitized_data.
        "ocr_total_estimate": parsed_ocr.TotalEstimate,
    }

    elapsed = time.perf_counter() - t0
    logger.info(
        "verification_agent | claim=%s | name_match=%s | policy_match=%s | "
        "warnings=%d | errors=%d | %.2fs (0 LLM calls)",
        claim_id, name_match, policy_match, len(warnings), len(errors), elapsed,
    )

    return {
        **state,
        "raw_data": raw_data,
        "sanitized_data": sanitized_data,
        "verification_output": verification_output.model_dump(),
        "customer_profile": {
            k: v for k, v in (customer or {}).items() if k != "_id"
        },
        "warnings": warnings,
        "errors": errors,
    }


# ── Sync wrapper ───────────────────────────────────────────────────────────────

def run_verification_agent(state: dict) -> dict:
    return asyncio.run(arun_verification_agent(state))