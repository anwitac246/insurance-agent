"""
policy_agent.py
---------------
Retrieves the relevant insurance policy via Pinecone RAG, performs semantic
exclusion matching, checks for aggregate limit breaches, and returns a
structured PolicyVerdict.

v3 — Root-cause fix for the 400 infinite-repetition loop
---------------------------------------------------------
The previous schema had two str fields with near-identical names right next
to each other:

    incident_covered: str   ← "Output 'yes' if covered, 'no' otherwise"
    exclusion_triggered: str ← "Output 'yes' if an exclusion applies, 'no' otherwise"

llama-3.1-8b on Groq enters a copy-paste loop filling these two fields
repeatedly until it hits the token limit, producing a malformed JSON blob
with hundreds of duplicate keys. Groq rejects it with a 400 tool_use_failed.

THE FIX: remove both boolean fields from the LLM schema entirely.

We already have everything needed to compute them deterministically:
  - exclusion_triggered: cosine similarity between narrative and exclusions
    text (already computed) + LLM's own exclusion_reason being non-empty.
  - incident_covered: NOT (exclusion_triggered OR aggregate_breach
    OR estimated_loss > remaining_limit).

The LLM now only produces fields it can fill reliably:
  - coverage_reasoning  (free-form text — no repetition risk)
  - exclusion_reason    (free-form text — empty string when no exclusion)

All boolean decisions are made in deterministic Python after the LLM returns.
This also makes the agent more reliable: the LLM can no longer approve a
claim that mathematically breaches the aggregate limit.
"""

import asyncio
import logging
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


# ── LLM output schema — NO boolean fields ─────────────────────────────────────
#
# Only free-text fields are requested from the LLM.
# Boolean decisions (incident_covered, exclusion_triggered) are computed
# deterministically in Python from the LLM's text output + numeric signals.
#
# Why: llama-3.1-8b on Groq enters an infinite copy-paste loop when the
# schema contains two near-identical str fields both described as 'yes/no'.
# Removing them eliminates the 400 tool_use_failed entirely.

class _PolicyReasoningRaw(BaseModel):
    """
    LLM-only fields. No booleans, no duplicated near-identical str fields.
    The model fills these reliably; everything else is computed in Python.
    """
    coverage_reasoning: str = Field(
        description=(
            "Step-by-step analysis: (1) does coverage_scope cover this incident? "
            "Quote the clause. (2) does any exclusion apply verbatim? "
            "Quote exact exclusion text if applicable. "
            "(3) is estimated_loss within remaining_limit? Show the arithmetic."
        )
    )
    exclusion_reason: str = Field(
        default="",
        description=(
            "If an exclusion clause directly and unambiguously applies to the "
            "narrative, quote the exact exclusion text here. "
            "Leave EMPTY if no exclusion applies — do NOT speculate."
        )
    )


# Public schema — used by downstream agents and metrics
class PolicyVerdict(BaseModel):
    policy_id: str
    policy_limit: float
    aggregate_limit: float
    deductible: float
    coverage_scope: str
    exclusions: str
    total_historical_payout: float
    remaining_limit: float
    incident_covered: bool        # computed deterministically
    exclusion_triggered: bool     # computed deterministically
    exclusion_reason: str
    coverage_reasoning: str


# ── Prompt — only requests reliable free-text fields ──────────────────────────

_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a senior insurance underwriter. Analyze the claim against the "
        "retrieved policy and provide detailed reasoning.\n\n"
        "Your response must follow these three numbered steps in coverage_reasoning:\n"
        "  1. Does coverage_scope include this incident type? Quote the relevant clause.\n"
        "  2. Does any exclusion apply verbatim to this narrative? "
        "     Quote the EXACT exclusion text if applicable.\n"
        "  3. Is estimated_loss within remaining_limit? State the arithmetic.\n\n"
        "For exclusion_reason:\n"
        "  - If an exclusion directly applies: quote the exact clause text.\n"
        "  - If no exclusion applies: leave it as an empty string.\n"
        "  - Do NOT infer or guess exclusions. Quote verbatim or leave empty.",
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
        "Provide coverage_reasoning (3 numbered steps) and exclusion_reason "
        "(exact quote or empty string).",
    ),
])


# ── Async LLM invocation — one call, two safe text fields only ────────────────

async def _ainvoke_llm(inputs: dict) -> _PolicyReasoningRaw:
    last_exc: Optional[Exception] = None

    for attempt in range(LLM_MAX_RETRIES):
        try:
            llm = get_async_llm()
            chain = _PROMPT | llm.with_structured_output(_PolicyReasoningRaw)
            raw: _PolicyReasoningRaw = await chain.ainvoke(inputs)
            record_success()
            return raw

        except Exception as exc:
            last_exc = exc
            exc_str = str(exc).lower()
            if "400" in exc_str or "bad request" in exc_str:
                logger.error(
                    "policy_agent | 400 Bad Request (non-retryable): %s", exc
                )
                raise
            if "429" in exc_str or "rate limit" in exc_str or "rate_limit" in exc_str:
                logger.warning(
                    "policy_agent | 429 detected (attempt %d)", attempt + 1
                )
                record_429()
                await asyncio.sleep(2 ** attempt)
            else:
                logger.warning(
                    "policy_agent | LLM error (attempt %d): %s", attempt + 1, exc
                )
                await asyncio.sleep(2 ** attempt)

    raise last_exc  # type: ignore[misc]


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
    result = index.query(
        vector=narrative_embedding, top_k=1, include_metadata=True
    )
    if result["matches"]:
        meta = result["matches"][0]["metadata"]
        if meta.get("policy_id") == policy_id:
            return meta
        logger.error(
            "Fallback returned policy %s — does not match expected %s. Refusing.",
            meta.get("policy_id"), policy_id,
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

    # ── Embed narrative ───────────────────────────────────────────────────────
    loop = asyncio.get_event_loop()
    try:
        narrative_vec = await loop.run_in_executor(None, _encode, narrative)
    except Exception as exc:
        errors.append(f"Policy agent: embedding failed — {exc}")
        return {**state, "policy_verdict": None, "errors": errors}

    # ── Fetch policy from Pinecone ────────────────────────────────────────────
    try:
        policy_meta = _fetch_policy(policy_id, narrative_vec.tolist())
    except Exception as exc:
        errors.append(f"Policy agent: Pinecone query failed — {exc}")
        return {**state, "policy_verdict": None, "errors": errors}

    if not policy_meta:
        errors.append(
            f"Policy agent: policy {policy_id} not found in vector store."
        )
        return {**state, "policy_verdict": None, "errors": errors}

    # ── Deterministic financial calculations ──────────────────────────────────
    aggregate_limit: float = float(policy_meta.get("aggregate_limit", 0))
    total_historical_payout: float = float(
        policy_meta.get("total_historical_payout", 0)
    )
    remaining_limit: float = aggregate_limit - total_historical_payout
    policy_limit: float = float(policy_meta.get("policy_limit", 0))
    deductible: float = float(policy_meta.get("deductible", 0))
    exclusions_text: str = policy_meta.get("exclusions", "")

    aggregate_breach = estimated_loss > remaining_limit
    if aggregate_breach:
        msg = (
            f"Aggregate limit breach: estimated loss ${estimated_loss:,.2f} "
            f"exceeds remaining limit ${remaining_limit:,.2f} "
            f"(aggregate=${aggregate_limit:,.2f}, "
            f"paid=${total_historical_payout:,.2f})."
        )
        errors.append(msg)
        logger.warning("claim=%s | %s", claim_id, msg)

    # ── Semantic exclusion similarity gate ────────────────────────────────────
    similarity = 0.0
    semantic_exclusion_signal = False
    try:
        exclusion_vec = await loop.run_in_executor(None, _encode, exclusions_text)
        similarity = float(
            cosine_similarity(
                narrative_vec.reshape(1, -1),
                exclusion_vec.reshape(1, -1),
            )[0][0]
        )
        semantic_exclusion_signal = similarity > EXCLUSION_SIMILARITY_THRESHOLD
        logger.debug(
            "claim=%s | exclusion similarity=%.4f | signal=%s",
            claim_id, similarity, semantic_exclusion_signal,
        )
    except Exception as exc:
        logger.warning("claim=%s | Exclusion similarity failed: %s", claim_id, exc)

    # ── LLM call — coverage_reasoning + exclusion_reason only ────────────────
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
        reasoning: _PolicyReasoningRaw = await _ainvoke_llm(llm_inputs)
    except Exception as exc:
        errors.append(
            f"Policy agent LLM failed after {LLM_MAX_RETRIES} retries: {exc}"
        )
        return {**state, "policy_verdict": None, "errors": errors}

    # ── Deterministic boolean decisions ───────────────────────────────────────
    #
    # exclusion_triggered:
    #   The LLM quoted an exclusion reason AND the semantic similarity gate
    #   confirms the narrative is actually close to the exclusions text.
    #   Requiring BOTH prevents the LLM from hallucinating exclusions AND
    #   prevents similarity alone from triggering on vague matches.
    #
    # incident_covered:
    #   False if exclusion fired, aggregate limit breached, or loss exceeds
    #   remaining limit. True otherwise.
    #   The LLM is not consulted for this boolean — it has all the information
    #   it needs to populate coverage_reasoning, and we derive the flag from
    #   hard arithmetic.

    exclusion_triggered = bool(reasoning.exclusion_reason) and semantic_exclusion_signal

    if not semantic_exclusion_signal and bool(reasoning.exclusion_reason):
        logger.info(
            "claim=%s | LLM exclusion_reason suppressed "
            "(similarity %.4f < threshold %.2f).",
            claim_id, similarity, EXCLUSION_SIMILARITY_THRESHOLD,
        )
        # Clear the reason since we're not triggering the exclusion
        reasoning = reasoning.model_copy(update={"exclusion_reason": ""})

    incident_covered = not (
        exclusion_triggered
        or aggregate_breach
        or estimated_loss > remaining_limit
    )

    # ── Append errors for triggered conditions ────────────────────────────────
    if exclusion_triggered:
        errors.append(
            f"Policy exclusion triggered: {reasoning.exclusion_reason}"
        )

    # ── Assemble final verdict ────────────────────────────────────────────────
    verdict = PolicyVerdict(
        policy_id=policy_meta.get("policy_id", policy_id),
        policy_limit=policy_limit,
        aggregate_limit=aggregate_limit,
        deductible=deductible,
        coverage_scope=policy_meta.get("coverage_scope", ""),
        exclusions=exclusions_text,
        total_historical_payout=total_historical_payout,
        remaining_limit=remaining_limit,
        incident_covered=incident_covered,
        exclusion_triggered=exclusion_triggered,
        exclusion_reason=reasoning.exclusion_reason,
        coverage_reasoning=reasoning.coverage_reasoning,
    )

    elapsed = time.perf_counter() - t_start
    logger.info(
        "policy_agent | claim=%s | done in %.2fs (1 LLM call) | "
        "covered=%s | exclusion=%s (sim=%.3f) | breach=%s",
        claim_id, elapsed, verdict.incident_covered,
        verdict.exclusion_triggered, similarity, aggregate_breach,
    )

    return {**state, "policy_verdict": verdict.model_dump(), "errors": errors}


# ── Sync wrapper ───────────────────────────────────────────────────────────────

def run_policy_agent(state: dict) -> dict:
    return asyncio.run(arun_policy_agent(state))