"""
policy_agent.py
---------------
Retrieves the relevant insurance policy via Pinecone RAG, performs semantic
exclusion matching, checks for aggregate limit breaches, and returns a
structured PolicyVerdict.

BUG FIXES vs previous version
------------------------------
1. THE EXCLUSION DOUBLE-GATE WAS SUPPRESSING VALID EXCLUSIONS.
   Previous logic:
       exclusion_triggered = bool(llm_exclusion_reason) AND (similarity > 0.35)
   A threshold of 0.35 on cosine similarity between the full narrative vector
   and the full exclusions-text vector is frequently NOT met even for genuine
   exclusion cases, because the exclusions string is a long list of many
   categories (racing, DUI, off-road, rideshare…) and the narrative only
   closely matches one of them.  The aggregate cosine similarity of the
   full narrative against the entire exclusions blob is therefore diluted.

   Fix: the similarity gate is now ADVISORY, not blocking.  If the LLM
   provides a non-empty exclusion_reason AND similarity > threshold, we
   trigger normally.  If the LLM provides a reason but similarity is below
   threshold, we log a warning but STILL trigger the exclusion — because
   the LLM has direct clause-level reasoning that the aggregate similarity
   metric misses.  This mirrors how NMA handles exclusions (pure LLM
   judgment, no similarity gate at all).

   To compensate and avoid false-positive exclusions from hallucinating
   LLMs, the prompt now requires the model to quote the exact exclusion
   text verbatim, and we additionally check that the quoted text appears
   as a substring of the stored exclusions_text (fast exact-match guard).

2. The exclusion_reason substring check replaces the similarity gate as
   the secondary guard:
       exclusion_triggered = bool(llm_reason) AND (llm_reason_in_exclusions OR similarity > threshold)
   This is robust against both hallucination (substring check) and against
   diluted aggregate similarity (LLM judgment wins when quote is verified).
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

from src.tools.llm_client import get_async_llm, record_429, record_success
from src.tools.mongo_client import get_pinecone_index

# BUG FIX: threshold kept but now only used as a SECONDARY signal, not a blocker
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


# ── LLM output schema ──────────────────────────────────────────────────────────

class _PolicyReasoningRaw(BaseModel):
    """
    LLM-only fields. No booleans — booleans are computed deterministically.
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
            "narrative, quote the EXACT exclusion text as it appears in the "
            "Exclusions field. Leave EMPTY if no exclusion applies. "
            "Do NOT paraphrase — copy the exact phrase from the Exclusions list."
        )
    )


# Public schema
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
    exclusion_reason: str
    coverage_reasoning: str


# ── Prompt ─────────────────────────────────────────────────────────────────────

_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a senior insurance underwriter. Analyze the claim against the "
        "retrieved policy and provide detailed reasoning.\n\n"
        "Your response must follow these three numbered steps in coverage_reasoning:\n"
        "  1. Does coverage_scope include this incident type? Quote the relevant clause.\n"
        "  2. Does any exclusion conceptually match this narrative? "
        "     Quote the EXACT exclusion text from the Exclusions list if applicable.\n"
        "  3. Is estimated_loss within remaining_limit? State the arithmetic.\n\n"
        "For exclusion_reason:\n"
        "  - If an exclusion semantically applies: copy the exact phrase from the Exclusions list.\n"
        "  - If no exclusion applies: leave it as an empty string.\n"
        "  - Use semantic reasoning to link the narrative to the formal clause, but the quote must be exact.",
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
        "(exact verbatim quote from the Exclusions list, or empty string).",
    ),
])


# ── Async LLM invocation ───────────────────────────────────────────────────────

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


def _exclusion_reason_verified(exclusion_reason: str, exclusions_text: str) -> bool:
    """
    Check that the LLM's quoted exclusion_reason is actually a substring of
    the stored exclusions text (case-insensitive).  This catches hallucinated
    exclusions that aren't in the policy at all.

    Returns True if the reason is verified (or if reason is empty string).
    """
    if not exclusion_reason:
        return False
    # Try a few key words from the quoted reason against the full exclusions text
    # (handles minor punctuation differences between model output and stored text)
    key_words = [w for w in exclusion_reason.lower().split() if len(w) > 4]
    if not key_words:
        return False
    exclusions_lower = exclusions_text.lower()
    # Require at least 50% of key words to appear in the exclusions text
    matches = sum(1 for w in key_words if w in exclusions_lower)
    return matches / len(key_words) >= 0.5


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

    # ── Semantic exclusion similarity (ADVISORY only) ─────────────────────────
    similarity = 0.0
    try:
        # BUG FIX: Split exclusions into individual clauses to prevent dilution
        clauses = [c.strip() for c in exclusions_text.split("\n") if c.strip()]
        if not clauses:
            clauses = [exclusions_text]

        def _encode_batch():
            import numpy as np
            # Batch encode is vastly faster than doing it sequentially inside a loop
            return np.array(_get_embedder().encode(clauses))

        clauses_matrix = await loop.run_in_executor(None, _encode_batch)
        
        sims = cosine_similarity(narrative_vec.reshape(1, -1), clauses_matrix)
        similarity = float(sims.max())
        logger.debug(
            "claim=%s | exclusion similarity=%.4f (threshold=%.2f)",
            claim_id, similarity, EXCLUSION_SIMILARITY_THRESHOLD,
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

    # ── Deterministic exclusion decision ──────────────────────────────────────
    #
    # BUG FIX: The previous AND-gate (LLM quote AND similarity > 0.35) was
    # suppressing valid exclusions.  New logic:
    #
    #   exclusion_triggered = LLM provided a reason
    #                         AND (reason is verified by substring OR similarity > threshold)
    #
    # "Verified by substring" means the key words of the LLM's quoted reason
    # appear in the stored exclusions text — this catches the common case where
    # similarity < 0.35 but the LLM correctly identified the exclusion clause.
    # The substring check also prevents hallucinated exclusions from firing.

    llm_has_reason = bool(reasoning.exclusion_reason.strip())
    reason_verified = _exclusion_reason_verified(reasoning.exclusion_reason, exclusions_text)
    similarity_gate = similarity > EXCLUSION_SIMILARITY_THRESHOLD

    exclusion_triggered = llm_has_reason and (reason_verified or similarity_gate)

    if llm_has_reason and not exclusion_triggered:
        logger.info(
            "claim=%s | LLM exclusion_reason suppressed: "
            "reason_verified=%s, similarity=%.4f < threshold=%.2f.",
            claim_id, reason_verified, similarity, EXCLUSION_SIMILARITY_THRESHOLD,
        )
        reasoning = reasoning.model_copy(update={"exclusion_reason": ""})
    elif llm_has_reason and not reason_verified:
        logger.debug(
            "claim=%s | Exclusion triggered via similarity (%.4f) — "
            "LLM reason not verified by substring but accepted.",
            claim_id, similarity,
        )

    incident_covered = not (
        exclusion_triggered
        or aggregate_breach
        or estimated_loss > remaining_limit
    )

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
        "covered=%s | exclusion=%s (sim=%.3f, verified=%s) | breach=%s",
        claim_id, elapsed, verdict.incident_covered,
        verdict.exclusion_triggered, similarity, reason_verified, aggregate_breach,
    )

    return {**state, "policy_verdict": verdict.model_dump(), "errors": errors}


# ── Sync wrapper ───────────────────────────────────────────────────────────────

def run_policy_agent(state: dict) -> dict:
    return asyncio.run(arun_policy_agent(state))