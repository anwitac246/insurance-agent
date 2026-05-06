from typing import Optional, TypedDict


class ClaimState(TypedDict):
   
    claim_id: str
    customer_profile: dict            # Customer_Profiles document
    raw_data: Optional[dict]          # Raw Active_Claims document
    sanitized_data: Optional[dict]    # Validated, normalised fields for downstream agents
    verification_output: Optional[dict]  # OcrExtraction + match flags + discrepancies

    policy_verdict: Optional[dict]

    fraud_report: Optional[dict]

    final_decision: Optional[dict]
    final_payout: Optional[float]

    errors: list[str]