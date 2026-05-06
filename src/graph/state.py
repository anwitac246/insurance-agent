from typing import Optional, TypedDict


class ClaimState(TypedDict):
    claim_id: str
    customer_profile: dict
    sanitized_data: Optional[dict]
    policy_verdict: Optional[dict]
    fraud_report: Optional[dict]
    final_decision: Optional[dict]
    final_payout: Optional[float]
    errors: list[str]