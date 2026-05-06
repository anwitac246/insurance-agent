import os
from enum import Enum
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from src.tools.mongo_client import get_pinecone_index

GROQ_MODEL = "llama-3.3-70b-versatile"
EXCLUSION_SIMILARITY_THRESHOLD = 0.35

_embedder = SentenceTransformer("all-MiniLM-L6-v2")
_llm = ChatGroq(model=GROQ_MODEL, temperature=0)


class PolicyVerdict(BaseModel):
    policy_id: str
    policy_limit: float
    aggregate_limit: float
    deductible: float
    coverage_scope: str
    exclusions: str
    total_historical_payout: float
    remaining_limit: float
    incident_covered: bool
    exclusion_triggered: bool
    exclusion_reason: str = ""
    coverage_reasoning: str


_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a senior insurance underwriter. Analyze the claim against the retrieved policy and "
        "determine coverage eligibility. Follow these steps explicitly in your reasoning:\n"
        "1. Does the coverage_scope include this type of incident? State yes/no and why.\n"
        "2. Does the exclusions clause apply to this incident narrative? Quote the relevant exclusion verbatim if so.\n"
        "3. Calculate remaining_limit = aggregate_limit - total_historical_payout. "
        "Is estimated_loss within remaining_limit?\n"
        "Set exclusion_triggered=True ONLY if the exclusion clause directly applies. "
        "Set incident_covered=False if exclusion_triggered OR estimated_loss > remaining_limit.",
    ),
    (
        "human",
        "=== Claim ===\n"
        "Claim ID: {claim_id}\n"
        "Incident Type: {incident_type}\n"
        "Narrative: {narrative}\n"
        "Estimated Loss: ${estimated_loss}\n\n"
        "=== Retrieved Policy ===\n"
        "Policy ID: {policy_id}\n"
        "Coverage Scope: {coverage_scope}\n"
        "Exclusions: {exclusions}\n"
        "Policy Limit: ${policy_limit}\n"
        "Aggregate Limit: ${aggregate_limit}\n"
        "Deductible: ${deductible}\n"
        "Total Historical Payout: ${total_historical_payout}\n"
        "Remaining Limit: ${remaining_limit}\n\n"
        "Provide a complete PolicyVerdict.",
    ),
])

_chain = _prompt | _llm.with_structured_output(PolicyVerdict)


def _fetch_policy(policy_id: str, narrative_embedding: list) -> dict | None:
    index = get_pinecone_index()

    result = index.query(
        vector=narrative_embedding,
        top_k=1,
        filter={"policy_id": {"$eq": policy_id}},
        include_metadata=True,
    )
    if result["matches"]:
        return result["matches"][0]["metadata"]

    result = index.query(vector=narrative_embedding, top_k=1, include_metadata=True)
    return result["matches"][0]["metadata"] if result["matches"] else None


def run_policy_agent(state: dict) -> dict:
    errors: list[str] = list(state.get("errors", []))
    sanitized = state.get("sanitized_data", {})

    narrative: str = sanitized.get("narrative", "")
    policy_id: str = sanitized.get("policy_id", "")
    estimated_loss: float = float(sanitized.get("estimated_loss", 0))

    narrative_embedding = _embedder.encode(narrative).tolist()

    policy_meta = _fetch_policy(policy_id, narrative_embedding)
    if not policy_meta:
        errors.append("Policy not found in vector store.")
        return {**state, "policy_verdict": None, "errors": errors}

    aggregate_limit = float(policy_meta.get("aggregate_limit", 0))
    total_historical_payout = float(policy_meta.get("total_historical_payout", 0))
    remaining_limit = aggregate_limit - total_historical_payout

    if estimated_loss > remaining_limit:
        errors.append(
            f"Aggregate limit breach: estimated loss ${estimated_loss:,.2f} "
            f"exceeds remaining limit ${remaining_limit:,.2f}."
        )

    exclusions_text: str = policy_meta.get("exclusions", "")
    exclusion_embedding = _embedder.encode(exclusions_text).reshape(1, -1)
    narrative_vec = _embedder.encode(narrative).reshape(1, -1)
    similarity = float(cosine_similarity(narrative_vec, exclusion_embedding)[0][0])
    semantic_exclusion_signal = similarity > EXCLUSION_SIMILARITY_THRESHOLD

    try:
        verdict: PolicyVerdict = _chain.invoke({
            "claim_id": sanitized.get("claim_id", ""),
            "incident_type": sanitized.get("incident_type", ""),
            "narrative": narrative,
            "estimated_loss": estimated_loss,
            "policy_id": policy_meta.get("policy_id", policy_id),
            "coverage_scope": policy_meta.get("coverage_scope", ""),
            "exclusions": exclusions_text,
            "policy_limit": policy_meta.get("policy_limit", 0),
            "aggregate_limit": aggregate_limit,
            "deductible": policy_meta.get("deductible", 0),
            "total_historical_payout": total_historical_payout,
            "remaining_limit": remaining_limit,
        })
    except Exception as exc:
        errors.append(f"Policy agent LLM failed: {exc}")
        return {**state, "policy_verdict": None, "errors": errors}

    if semantic_exclusion_signal and verdict.exclusion_triggered:
        verdict = verdict.model_copy(update={"exclusion_triggered": True, "incident_covered": False})

    if verdict.exclusion_triggered:
        errors.append(f"Policy exclusion triggered: {verdict.exclusion_reason}")

    return {**state, "policy_verdict": verdict.model_dump(), "errors": errors}