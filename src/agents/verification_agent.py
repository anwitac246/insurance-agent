from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic
from src.tools.mongo_client import get_db


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


_llm = ChatAnthropic(model="claude-3-5-haiku-20241022", temperature=0)
_structured_llm = _llm.with_structured_output(OcrExtraction)

_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a document verification specialist for car insurance claims. "
        "Extract and validate the OCR fields exactly as they appear. Do not infer missing values.",
    ),
    (
        "human",
        "Parse and validate the following OCR extraction from an insurance claim document:\n\n"
        "{ocr_json}\n\n"
        "Return the structured fields.",
    ),
])

_chain = _prompt | _structured_llm


def run_verification_agent(state: dict) -> dict:
    db = get_db()
    errors: list[str] = list(state.get("errors", []))

    claim = db["Active_Claims"].find_one({"claim_id": state["claim_id"]})
    if not claim:
        return {**state, "errors": errors + [f"Claim {state['claim_id']} not found in Active_Claims."]}

    raw_data = {k: v for k, v in claim.items() if k != "_id"}

    ocr_raw = claim.get("ocr_extraction", {})
    try:
        parsed_ocr: OcrExtraction = _chain.invoke({"ocr_json": str(ocr_raw)})
    except Exception as exc:
        return {**state, "raw_data": raw_data, "errors": errors + [f"OCR parsing failed: {exc}"]}

    customer = db["Customer_Profiles"].find_one({"customer_id": claim["customer_id"]})
    discrepancies: list[str] = []

    name_match = False
    policy_match = False

    if customer:
        stored_name: str = customer.get("full_name", "")
        stored_policy: str = customer.get("policy_id", "")

        name_match = parsed_ocr.ClaimantName.strip().lower() == stored_name.strip().lower()
        policy_match = parsed_ocr.PolicyNumber.strip().upper() == stored_policy[:8].upper()

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

    return {
        **state,
        "raw_data": raw_data,
        "verification_output": verification_output,
        "errors": errors,
    }