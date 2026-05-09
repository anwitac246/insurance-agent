"""
policy_agent.py
---------------
Retrieves the relevant insurance policy via Pinecone RAG, performs semantic
exclusion matching, checks for aggregate limit breaches, and returns a
structured PolicyVerdict.

Performance fix (single LLM call)
----------------------------------
Previously used a two-step approach: one structured-output call (without
boolean fields) + one plain-text call to extract booleans. That doubled
LLM usage. Now uses a single structured-output call where the two boolean
fields (`incident_covered`, `exclusion_triggered`) are typed as `str`.
Pydantic v2 coerces any JSON value (True, False, "yes", "no") to str,
eliminating the Groq 400 Bad Request errors caused by Literal["yes","no"]
constraints, while keeping the number of LLM calls at 1.
"""

import asyncio
import logging
import re
import time
from functools import lru_cache
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.tools.groq_client import get_async_llm, record_429, record_success
from src.tools.mongo_client import get_pinecone_index

EXCLUSION_SIMILARITY_THRESHOLD = 0.35
EMBEDDER_MODEL = "all-MiniLM-L6-v2"
LLM_MAX_RETRIES = 3

logger = logging.getLogger(__name__)


# ── Cached embedder singleton ──────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_embedder() -> SentenceTransformer:
    logger.info("Loading SentenceTransformer model: %s", EMBEDDER_MODEL)
    return SentenceTransformer(EMBEDDER_MODEL)


@lru_cache(maxsize=512)
def _cached_encode(text: str) -> tuple:
    vec = _get_embedder().encode(text)
    return tuple(vec.tolist())


def _encode(text: str):
    import numpy as np
    return np.array(_cached_encode(text))


# ── Output schemas ─────────────────────────────────────────────────────────────
# boolean fields typed as `str` — Pydantic v2 coerces True/False/"yes"/"no"
# to str without raising, avoiding Groq 400 tool_use_failed errors.

class _PolicyVerdictRaw(BaseModel):
    policy_id: str
    policy_limit: float
    aggregate_limit: float
    deductible: float
    coverage_scope: str
    exclusions: str
    total_historical_payout: float
    remaining_limit: float
    incident_covered: str = Field(
        description='Output "yes" if the incident is covered, "no" otherwise.'
    )
    exclusion_triggered: str = Field(
        description='Output "yes" if a policy exclusion applies, "no" otherwise.'
    )
    exclusion_reason: str = ""
    coverage_reasoning: str


def _parse_bool(val: str) -> bool:
    return str(val).lower().strip() in ("yes", "true", "1")


# Public schema — proper booleans, used by downstream agents and metrics
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
        "  - Set exclusion_triggered to 'yes' ONLY if an exclusion clause directly and "
        "unambiguously applies to the narrative.\n"
        "  - Set incident_covered to 'no' if exclusion_triggered is 'yes' OR "
        "estimated_loss > remaining_limit.\n"
        "  - Do NOT infer or guess exclusions — quote verbatim or set exclusion_triggered to 'no'.\n"
        "  - For incident_covered and exclusion_triggered output the string 'yes' or 'no'.",
    ),
    (
        "human",
        "=== Claim ===\n"
        "Claim ID        : {claim_id}\n"
        "Incident Type   : {incident_type}\n"
        "Narrative       : {narrative}\n"
        "Estimated Loss  : ${estimated_loss}\n\n"
        "=== Retrieved Policy ===\n"
        "Policy ID               : {policy_id}\n"
        "Coverage Scope          : {coverage_scope}\n"
        "Exclusions              : {exclusions}\n"
        "Policy Limit            : ${policy_limit}\n"
        "Aggregate Limit         : ${aggregate_limit}\n"
        "Deductible              : ${deductible}\n"
        "Total Historical Payout : ${total_historical_payout}\n"
        "Remaining Limit         : ${remaining_limit}\n\n"
        "Provide a complete PolicyVerdict with all fields.",
    ),
])


# ── Async LLM invocation (single call) ────────────────────────────────────────

async def _ainvoke_llm(inputs: dict) -> PolicyVerdict:
    last_exc = None

    for attempt in range(LLM_MAX_RETRIES):
        try:
            llm = get_async_llm()
            chain = _PROMPT | llm.with_structured_output(_PolicyVerdictRaw)
            raw: _PolicyVerdictRaw = await chain.ainvoke(inputs)
            record_success()
            return PolicyVerdict(
                policy_id=raw.policy_id,
                policy_limit=raw.policy_limit,
                aggregate_limit=raw.aggregate_limit,
                deductible=raw.deductible,
                coverage_scope=raw.coverage_scope,
                exclusions=raw.exclusions,
                total_historical_payout=raw.total_historical_payout,
                remaining_limit=raw.remaining_limit,
                incident_covered=_parse_bool(raw.incident_covered),
                exclusion_triggered=_parse_bool(raw.exclusion_triggered),
                exclusion_reason=raw.exclusion_reason,
                coverage_reasoning=raw.coverage_reasoning,
            )

        except Exception as exc:
            last_exc = exc
            exc_str = str(exc).lower()
            if "400" in exc_str or "bad request" in exc_str:
                logger.error("policy_agent | 400 Bad Request (non-retryable): %s", exc)
                raise
            if "429" in exc_str or "rate limit" in exc_str or "rate_limit" in exc_str:
                logger.warning("policy_agent | 429 detected (attempt %d)", attempt + 1)
                record_429()
                await asyncio.sleep(2 ** attempt)
            else:
                logger.warning("policy_agent | LLM error (attempt %d): %s", attempt + 1, exc)
                await asyncio.sleep(2 ** attempt)

    raise last_exc


# ── Pinecone retrieval ─────────────────────────────────────────────────────────

def _fetch_policy(policy_id: str, narrative_embedding: list) -> Optional[dict]:
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
        "Exact policy filter returned no results for %s. Trying unfiltered fallback.",
        policy_id,
    )
    result = index.query(vector=narrative_embedding, top_k=1, include_metadata=True)
    if result["matches"]:
        meta = result["matches"][0]["metadata"]
        if meta.get("policy_id") == policy_id:
            return meta
        logger.error(
            "Fallback returned policy %s — does not match expected %s. Refusing.",
            meta.get("policy_id"),
            policy_id,
        )
    return None


# ── Async main ─────────────────────────────────────────────────────────────────

async def arun_policy_agent(state: dict) -> dict:
    t_start = time.perf_counter()
    errors: list[str] = list(state.get("errors", []))
    sanitized: dict = state.get("sanitized_data", {})

    claim_id: str = sanitized.get("claim_id", "unknown")
    narrative: str = sanitized.get("narrative", "")
    policy_id: str = sanitized.get("policy_id", "")
    estimated_loss: float = float(sanitized.get("estimated_loss", 0))

    logger.info(
        "policy_agent | claim=%s | policy=%s | loss=%.2f",
        claim_id, policy_id, estimated_loss,
    )

    if not policy_id:
        errors.append("Policy agent: policy_id missing from sanitized_data.")
        return {**state, "policy_verdict": None, "errors": errors}

    if not narrative:
        errors.append("Policy agent: narrative missing from sanitized_data.")
        return {**state, "policy_verdict": None, "errors": errors}

    loop = asyncio.get_event_loop()
    try:
        narrative_vec = await loop.run_in_executor(None, _encode, narrative)
    except Exception as exc:
        errors.append(f"Policy agent: embedding failed — {exc}")
        return {**state, "policy_verdict": None, "errors": errors}

    try:
        policy_meta = _fetch_policy(policy_id, narrative_vec.tolist())
    except Exception as exc:
        errors.append(f"Policy agent: Pinecone query failed — {exc}")
        return {**state, "policy_verdict": None, "errors": errors}

    if not policy_meta:
        errors.append(f"Policy agent: policy {policy_id} not found in vector store.")
        return {**state, "policy_verdict": None, "errors": errors}

    # ── Deterministic financial calculations ───────────────────────────────────
    aggregate_limit: float = float(policy_meta.get("aggregate_limit", 0))
    total_historical_payout: float = float(policy_meta.get("total_historical_payout", 0))
    remaining_limit: float = aggregate_limit - total_historical_payout

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
    semantic_exclusion_signal = False
    similarity = 0.0
    try:
        exclusion_vec = await loop.run_in_executor(None, _encode, exclusions_text)
        similarity = float(
            cosine_similarity(
                narrative_vec.reshape(1, -1), exclusion_vec.reshape(1, -1)
            )[0][0]
        )
        semantic_exclusion_signal = similarity > EXCLUSION_SIMILARITY_THRESHOLD
        logger.debug(
            "claim=%s | exclusion similarity=%.4f | signal=%s",
            claim_id, similarity, semantic_exclusion_signal,
        )
    except Exception as exc:
        logger.warning("claim=%s | Exclusion similarity failed: %s", claim_id, exc)

    # ── LLM adjudication (single call) ────────────────────────────────────────
    policy_limit = float(policy_meta.get("policy_limit", 0))
    deductible = float(policy_meta.get("deductible", 0))

    llm_inputs = {
        "claim_id": claim_id,
        "incident_type": sanitized.get("incident_type", ""),
        "narrative": narrative,
        "estimated_loss": f"{estimated_loss:,.2f}",
        "policy_id": policy_meta.get("policy_id", policy_id),
        "coverage_scope": policy_meta.get("coverage_scope", ""),
        "exclusions": exclusions_text,
        "policy_limit": f"{policy_limit:,.2f}",
        "aggregate_limit": f"{aggregate_limit:,.2f}",
        "deductible": f"{deductible:,.2f}",
        "total_historical_payout": f"{total_historical_payout:,.2f}",
        "remaining_limit": f"{remaining_limit:,.2f}",
    }

    try:
        verdict: PolicyVerdict = await _ainvoke_llm(llm_inputs)
        logger.info(
            "claim=%s | LLM verdict: covered=%s exclusion=%s",
            claim_id, verdict.incident_covered, verdict.exclusion_triggered,
        )
    except Exception as exc:
        errors.append(f"Policy agent LLM failed after {LLM_MAX_RETRIES} retries: {exc}")
        return {**state, "policy_verdict": None, "errors": errors}

    # ── Post-LLM deterministic overrides ──────────────────────────────────────
    if not semantic_exclusion_signal and verdict.exclusion_triggered:
        logger.info(
            "claim=%s | LLM exclusion suppressed (similarity %.4f < threshold %.2f).",
            claim_id, similarity, EXCLUSION_SIMILARITY_THRESHOLD,
        )
        verdict = verdict.model_copy(
            update={"exclusion_triggered": False, "exclusion_reason": ""}
        )

    overrides: dict = {"remaining_limit": remaining_limit}
    if aggregate_breach or verdict.exclusion_triggered:
        overrides["incident_covered"] = False

    verdict = verdict.model_copy(update=overrides)

    if verdict.exclusion_triggered:
        errors.append(f"Policy exclusion triggered: {verdict.exclusion_reason}")

    elapsed = time.perf_counter() - t_start
    logger.info(
        "policy_agent | claim=%s | done in %.2fs | covered=%s | exclusion=%s | breach=%s",
        claim_id, elapsed, verdict.incident_covered,
        verdict.exclusion_triggered, aggregate_breach,
    )

    return {**state, "policy_verdict": verdict.model_dump(), "errors": errors}


# ── Sync wrapper ───────────────────────────────────────────────────────────────

def run_policy_agent(state: dict) -> dict:
    return asyncio.run(arun_policy_agent(state))