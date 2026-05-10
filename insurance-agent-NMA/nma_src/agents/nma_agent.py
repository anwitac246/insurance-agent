import sys
import os
import asyncio
import logging
from langchain_core.prompts import ChatPromptTemplate

from nma_src.schemas.nma_schema import NMAOutputRaw, NMAOutput
from src.tools.groq_client import get_async_llm, record_429, record_success

logger = logging.getLogger(__name__)

_PROMPT = ChatPromptTemplate.from_messages([
    ("system", 
     "You are an end-to-end insurance claim adjudicator. Analyze fraud, check policy coverage, and calculate the payout.\n"
     "Payout formula: min(Estimated_Loss - Deductible, Remaining_Limit).\n"
     "Deny if fraud risk is High, exclusion is triggered, or not covered.\n"
     "Output strictly the requested fields."
    ),
    ("human", 
     "=== Claim Data ===\n{claim}\n\n"
     "=== Customer Data ===\n{customer}\n\n"
     "=== Policy Data ===\n{policy}"
    )
])

async def arun_nma_agent(context: dict):
    claim_str = str(context["claim"])
    cust_str = str(context["customer"])
    pol_str = str(context["policy"])
    
    for attempt in range(3):
        try:
            llm = get_async_llm()
            chain = _PROMPT | llm.with_structured_output(NMAOutputRaw)
            raw = await chain.ainvoke({"claim": claim_str, "customer": cust_str, "policy": pol_str})
            record_success()
            
            return NMAOutput(
                fraud_risk_score=raw.fraud_risk_score,
                fraud_anomalies=raw.fraud_anomalies,
                incident_covered=raw.incident_covered.lower() in ("yes", "true", "1"),
                exclusion_triggered=raw.exclusion_triggered.lower() in ("yes", "true", "1"),
                exclusion_reason=raw.exclusion_reason,
                final_payout=float(raw.final_payout),
                approved=raw.approved.lower() in ("yes", "true", "1"),
                step_by_step_reasoning=raw.step_by_step_reasoning
            )
        except Exception as exc:
            if "429" in str(exc).lower() or "rate limit" in str(exc).lower():
                record_429()
                await asyncio.sleep(2**attempt)
            elif "400" in str(exc).lower():
                raise
            else:
                await asyncio.sleep(2**attempt)
    raise RuntimeError("NMA Agent failed after 3 retries")
