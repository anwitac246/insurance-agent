"""
verification_agent.py
---------------------
Document verification agent.

Migrated from ChatAnthropic (claude-3-5-haiku) → ChatGroq (llama-3.3-70b-versatile).
No Anthropic API key is required anywhere in this project.

Responsibilities
----------------
  1. Fetch the Active_Claim record from MongoDB.
  2. Parse the OCR extraction dict into a validated OcrExtraction Pydantic model.
  3. Cross-check ClaimantName and PolicyNumber against the Customer_Profiles collection.
  4. Populate state with raw_data, verification_output, and sanitized_data.
"""

from __future__ import annotations

import logging
import time
from functools import lru_cache
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.tools.mongo_client import get_db

logger = logging.getLogger(__name__)

GROQ_MODEL = "llama-3.3-70b-versatile"
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


# ── LLM singleton ──────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_llm() -> ChatGroq:
    return ChatGroq(model=GROQ_MODEL, temperature=0, request_timeout=30)


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


@retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(LLM_MAX_RETRIES),
    wait=wait_exponential(min=2, max=10),
    reraise=True,
)
def _invoke_ocr(chain, inputs: dict) -> OcrExtraction:
    return chain.invoke(inputs)


# ── Main agent function ────────────────────────────────────────────────────────

def run_verification_agent(state: dict) -> dict:
    t0 = time.perf_counter()
    db = get_db()
    errors: list[str] = list(state.get("errors", []))
    claim_id: str = state.get("claim_id", "")

    # ── Fetch claim ────────────────────────────────────────────────────────────
    claim = db["Active_Claims"].find_one({"claim_id": claim_id})
    if not claim:
        return {
            **state,
            "errors": errors + [f"Claim {claim_id} not found in Active_Claims."],
        }

    raw_data = {k: v for k, v in claim.items() if k != "_id"}

    # ── OCR parsing ────────────────────────────────────────────────────────────
    ocr_raw = claim.get("ocr_extraction", {})
    chain = _prompt | _get_llm().with_structured_output(OcrExtraction)
    try:
        parsed_ocr: OcrExtraction = _invoke_ocr(chain, {"ocr_json": str(ocr_raw)})
    except Exception as exc:
        logger.exception("OCR parsing failed for claim %s", claim_id)
        return {
            **state,
            "raw_data": raw_data,
            "errors": errors + [f"OCR parsing failed: {exc}"],
        }

    # ── Cross-check against Customer_Profiles ─────────────────────────────────
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

    # ── Build sanitized_data for downstream agents ─────────────────────────────
    # This centralised dict is the single source of truth for all agents.
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