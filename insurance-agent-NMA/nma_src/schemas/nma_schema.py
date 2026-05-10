from pydantic import BaseModel, Field

class NMAOutputRaw(BaseModel):
    fraud_risk_score: int = Field(ge=1, le=10, description="1-10 fraud risk score")
    fraud_anomalies: str = Field(description="List anomalies or 'None'")
    incident_covered: str = Field(description="'yes' or 'no'")
    exclusion_triggered: str = Field(description="'yes' or 'no'")
    exclusion_reason: str = Field(description="Exact quote or empty")
    final_payout: str = Field(description="Calculated payout as a string. Example: '0.0' or '2213.65'")
    approved: str = Field(description="'yes' or 'no'")
    step_by_step_reasoning: str = Field(description="MAX 1 SENTENCE reasoning.")

class NMAOutput(BaseModel):
    fraud_risk_score: int
    fraud_anomalies: str
    incident_covered: bool
    exclusion_triggered: bool
    exclusion_reason: str
    final_payout: float
    approved: bool
    step_by_step_reasoning: str
