"""
policy_agent.py
---------------
Retrieves the relevant insurance policy via Pinecone RAG, performs semantic
exclusion matching, checks for aggregate limit breaches, and returns a
structured PolicyVerdict.

All financial calculations are deterministic (Python) — never delegated to
the LLM.  The dual-gate exclusion logic requires BOTH a cosine-similarity
signal AND LLM confirmation before an exclusion is triggered.
"""

import logging
import time
from functools import lru_cache
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.tools.mongo_client import get_pinecone_index

# ── Constants ──────────────────────────────────────────────────────────────────
GROQ_MODEL = "llama-3.3-70b-versatile"
EXCLUSION_SIMILARITY_THRESHOLD = 0.35
EMBEDDER_MODEL = "all-MiniLM-L6-v2"
LLM_MAX_RETRIES = 3
LLM_RETRY_WAIT_MIN = 2   # seconds
LLM_RETRY_WAIT_MAX = 10  # seconds

# ── Logging ────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── Lazy singletons ────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _get_embedder() -> SentenceTransformer:
    logger.info("Loading SentenceTransformer model: %s", EMBEDDER_MODEL)
    return SentenceTransformer(EMBEDDER_MODEL)


@lru_cache(maxsize=1)
def _get_llm() -> ChatGroq:
    return ChatGroq(model=GROQ_MODEL, temperature=0, request_timeout=30)


# ── Output schema ──────────────────────────────────────────────────────────────
class PolicyVerdict(BaseModel):
    policy_id: str
    policy_limit: float
    aggregate_limit: float
    deductible: float
    coverage_scope: str
    exclusions: str
    total_historical_payout: float
    remaining_limit: float          # always overridden with Python-computed value post-LLM
    incident_covered: bool
    exclusion_triggered: bool
    exclusion_reason: str = ""
    coverage_reasoning: str


# ── Prompt ─────────────────────────────────────────────────────────────────────
_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a senior insurance underwriter. Analyze the claim against the retrieved policy.\n"
        "Follow these numbered steps explicitly in coverage_reasoning:\n"
        "  1. Does coverage_scope include this incident type? Quote the relevant clause.\n"
        "  2. Does the exclusions clause apply verbatim to this narrative? "
        "Quote the exact exclusion text if applicable.\n"
        "  3. Is estimated_loss within remaining_limit? State the arithmetic.\n\n"
        "Rules:\n"
        "  - Set exclusion_triggered=True ONLY if an exclusion clause directly and "
        "unambiguously applies to the narrative.\n"
        "  - Set incident_covered=False if exclusion_triggered OR "
        "estimated_loss > remaining_limit.\n"
        "  - Do NOT infer or guess exclusions — quote verbatim or leave exclusion_triggered=False.",
    ),
    (
        "human",
        "=== Claim ===\n"
        "Claim ID        : {claim_id}\n"
        "Incident Type   : {incident_type}\n"
        "Narrative       : {narrative}\n"
        "Estimated Loss  : ${estimated_loss:,.2f}\n\n"
        "=== Retrieved Policy ===\n"
        "Policy ID               : {policy_id}\n"
        "Coverage Scope          : {coverage_scope}\n"
        "Exclusions              : {exclusions}\n"
        "Policy Limit            : ${policy_limit:,.2f}\n"
        "Aggregate Limit         : ${aggregate_limit:,.2f}\n"
        "Deductible              : ${deductible:,.2f}\n"
        "Total Historical Payout : ${total_historical_payout:,.2f}\n"
        "Remaining Limit         : ${remaining_limit:,.2f}\n\n"
        "Provide a complete PolicyVerdict.",
    ),
])


# ── Retry-wrapped LLM call ─────────────────────────────────────────────────────
@retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(LLM_MAX_RETRIES),
    wait=wait_exponential(min=LLM_RETRY_WAIT_MIN, max=LLM_RETRY_WAIT_MAX),
    reraise=True,
)
def _invoke_llm(chain, inputs: dict) -> PolicyVerdict:
    return chain.invoke(inputs)


# ── Pinecone retrieval ─────────────────────────────────────────────────────────
def _fetch_policy(policy_id: str, narrative_embedding: list) -> Optional[dict]:
    """
    Primary  : exact policy_id filter.
    Fallback : top-1 unfiltered — accepted only if policy_id still matches,
               preventing adjudication against a different customer's policy.
    """
    index = get_pinecone_index()

    result = index.query(
        vector=narrative_embedding,
        top_k=1,
        filter={"policy_id": {"$eq": policy_id}},
        include_metadata=True,
    )
    if result["matches"]:
        logger.debug("Policy %s retrieved via exact filter.", policy_id)
        return result["matches"][0]["metadata"]

    logger.warning(
        "Exact policy filter returned no results for %s. Trying unfiltered fallback.", policy_id
    )
    result = index.query(vector=narrative_embedding, top_k=1, include_metadata=True)
    if result["matches"]:
        meta = result["matches"][0]["metadata"]
        if meta.get("policy_id") == policy_id:
            logger.info("Fallback matched correct policy %s.", policy_id)
            return meta
        logger.error(
            "Fallback returned policy %s — does not match expected %s. Refusing to use it.",
            meta.get("policy_id"),
            policy_id,
        )

    return None


# ── Main agent function ────────────────────────────────────────────────────────
def run_policy_agent(state: dict) -> dict:
    """
    Populates state["policy_verdict"] with a PolicyVerdict dict.
    Appends to state["errors"] on any detected issue.
    Never raises — always returns a valid state dict.
    """
    t_start = time.perf_counter()
    errors: list[str] = list(state.get("errors", []))
    sanitized: dict = state.get("sanitized_data", {})

    claim_id: str = sanitized.get("claim_id", "unknown")
    narrative: str = sanitized.get("narrative", "")
    policy_id: str = sanitized.get("policy_id", "")
    estimated_loss: float = float(sanitized.get("estimated_loss", 0))

    logger.info("policy_agent | claim=%s | policy=%s | loss=%.2f", claim_id, policy_id, estimated_loss)

    if not policy_id:
        errors.append("Policy agent: policy_id missing from sanitized_data.")
        return {**state, "policy_verdict": None, "errors": errors}

    if not narrative:
        errors.append("Policy agent: narrative missing from sanitized_data.")
        return {**state, "policy_verdict": None, "errors": errors}

    # ── Embeddings ─────────────────────────────────────────────────────────────
    embedder = _get_embedder()
    try:
        narrative_vec = embedder.encode(narrative)          # shape (384,)
    except Exception as exc:
        errors.append(f"Policy agent: embedding failed — {exc}")
        logger.exception("Embedding failed for claim %s", claim_id)
        return {**state, "policy_verdict": None, "errors": errors}

    # ── Policy retrieval ───────────────────────────────────────────────────────
    try:
        policy_meta = _fetch_policy(policy_id, narrative_vec.tolist())
    except Exception as exc:
        errors.append(f"Policy agent: Pinecone query failed — {exc}")
        logger.exception("Pinecone query failed for claim %s", claim_id)
        return {**state, "policy_verdict": None, "errors": errors}

    if not policy_meta:
        errors.append(f"Policy agent: policy {policy_id} not found in vector store.")
        return {**state, "policy_verdict": None, "errors": errors}

    # ── Deterministic financial calculations ───────────────────────────────────
    aggregate_limit: float = float(policy_meta.get("aggregate_limit", 0))
    total_historical_payout: float = float(policy_meta.get("total_historical_payout", 0))
    remaining_limit: float = aggregate_limit - total_historical_payout  # always Python-computed

    aggregate_breach = estimated_loss > remaining_limit
    if aggregate_breach:
        msg = (
            f"Aggregate limit breach: estimated loss ${estimated_loss:,.2f} "
            f"exceeds remaining limit ${remaining_limit:,.2f} "
            f"(aggregate=${aggregate_limit:,.2f}, paid=${total_historical_payout:,.2f})."
        )
        errors.append(msg)
        logger.warning("claim=%s | %s", claim_id, msg)

    # ── Semantic exclusion gate ────────────────────────────────────────────────
    exclusions_text: str = policy_meta.get("exclusions", "")
    try:
        exclusion_vec = embedder.encode(exclusions_text).reshape(1, -1)
        similarity: float = float(
            cosine_similarity(narrative_vec.reshape(1, -1), exclusion_vec)[0][0]
        )
        semantic_exclusion_signal = similarity > EXCLUSION_SIMILARITY_THRESHOLD
        logger.debug(
            "claim=%s | exclusion similarity=%.4f | signal=%s",
            claim_id, similarity, semantic_exclusion_signal,
        )
    except Exception as exc:
        logger.warning("claim=%s | Exclusion similarity failed (defaulting to False): %s", claim_id, exc)
        semantic_exclusion_signal = False
        similarity = 0.0

    # ── LLM adjudication ──────────────────────────────────────────────────────
    chain = _PROMPT | _get_llm().with_structured_output(PolicyVerdict)
    llm_inputs = {
        "claim_id": claim_id,
        "incident_type": sanitized.get("incident_type", ""),
        "narrative": narrative,
        "estimated_loss": estimated_loss,
        "policy_id": policy_meta.get("policy_id", policy_id),
        "coverage_scope": policy_meta.get("coverage_scope", ""),
        "exclusions": exclusions_text,
        "policy_limit": float(policy_meta.get("policy_limit", 0)),
        "aggregate_limit": aggregate_limit,
        "deductible": float(policy_meta.get("deductible", 0)),
        "total_historical_payout": total_historical_payout,
        "remaining_limit": remaining_limit,
    }

    try:
        verdict: PolicyVerdict = _invoke_llm(chain, llm_inputs)
        logger.info(
            "claim=%s | LLM verdict: covered=%s exclusion=%s",
            claim_id, verdict.incident_covered, verdict.exclusion_triggered,
        )
    except Exception as exc:
        errors.append(f"Policy agent LLM failed after {LLM_MAX_RETRIES} retries: {exc}")
        logger.exception("LLM failed for claim %s", claim_id)
        return {**state, "policy_verdict": None, "errors": errors}

    # ── Post-LLM deterministic overrides ──────────────────────────────────────
    # Rule 1: Dual-gate — suppress LLM exclusion if semantic similarity is insufficient.
    #         The LLM alone cannot trigger an exclusion; semantic evidence is required.
    if not semantic_exclusion_signal and verdict.exclusion_triggered:
        logger.info(
            "claim=%s | LLM exclusion suppressed (similarity %.4f < threshold %.2f).",
            claim_id, similarity, EXCLUSION_SIMILARITY_THRESHOLD,
        )
        verdict = verdict.model_copy(update={"exclusion_triggered": False, "exclusion_reason": ""})

    # Rule 2: Always use Python-computed remaining_limit — never trust LLM arithmetic.
    # Rule 3: Force incident_covered=False if aggregate breach or exclusion confirmed.
    overrides: dict = {"remaining_limit": remaining_limit}
    if aggregate_breach or verdict.exclusion_triggered:
        overrides["incident_covered"] = False

    verdict = verdict.model_copy(update=overrides)

    if verdict.exclusion_triggered:
        errors.append(f"Policy exclusion triggered: {verdict.exclusion_reason}")

    elapsed = time.perf_counter() - t_start
    logger.info(
        "policy_agent | claim=%s | done in %.2fs | covered=%s | exclusion=%s | breach=%s",
        claim_id, elapsed, verdict.incident_covered, verdict.exclusion_triggered, aggregate_breach,
    )

    return {**state, "policy_verdict": verdict.model_dump(), "errors": errors}