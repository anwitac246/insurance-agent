"""
context_fetcher.py
------------------
Fetches all context needed by the NMA single agent.
"""

import asyncio
import logging
from typing import Optional

from sentence_transformers import SentenceTransformer
from src.tools.mongo_client import get_db, get_pinecone_index

logger = logging.getLogger(__name__)

_embedder: Optional[SentenceTransformer] = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if not _embedder:
        logger.info("Loading SentenceTransformer model: all-MiniLM-L6-v2")
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def _summarize_history(records: list[dict], max_records: int = 10) -> str:
    """Produce a plain-text summary of past claims for the LLM prompt."""
    if not records:
        return "No prior claim history found."

    recent = sorted(records, key=lambda r: r.get("incident_date", ""), reverse=True)
    shown = recent[:max_records]

    denied_or_flagged = sum(
        1 for r in records if r.get("claim_status") in ("Denied", "Fraud_Flagged")
    )
    total_paid = sum(r.get("payout_amount", 0.0) for r in records if r.get("claim_status") == "Approved")

    lines = [
        f"Total historical claims: {len(records)} "
        f"(denied/flagged: {denied_or_flagged}, total approved payout: ${total_paid:,.2f})",
        f"(Showing most recent {len(shown)} records)",
    ]
    for r in shown:
        lines.append(
            f"  [{r.get('incident_date', 'N/A')}] {r.get('incident_type', '?')} | "
            f"Status: {r.get('claim_status', '?')} | "
            f"Payout: ${r.get('payout_amount', 0.0):,.2f}"
        )
    return "\n".join(lines)


async def fetch_nma_context(claim_id: str) -> dict:
    """
    Fetch everything the single NMA agent needs to adjudicate a claim:
      - claim document       (from Active_Claims)
      - customer profile     (from Customer_Profiles)
      - claim history        (from Claim_History)  ← new
      - policy metadata      (from Pinecone via RAG)
    """
    db = get_db()

    # ── Claim ──────────────────────────────────────────────────────────────────
    # BUG FIX: was db.claims (wrong collection) → db["Active_Claims"]
    claim = await asyncio.to_thread(
        db["Active_Claims"].find_one, {"claim_id": claim_id}
    )
    if not claim:
        raise ValueError(
            f"Claim {claim_id} not found in Active_Claims. "
            "Make sure seed_data.py has been run."
        )

    # Strip internal Mongo _id before passing to LLM
    claim = {k: v for k, v in claim.items() if k != "_id"}

    customer_id: str = claim.get("customer_id", "")
    policy_id: str = claim.get("policy_id", "")

    # BUG FIX: field was "incident_narrative" → correct name is "narrative"
    narrative: str = claim.get("narrative", "")

    # ── Customer profile ───────────────────────────────────────────────────────
    # BUG FIX: was db.customers (wrong collection) → db["Customer_Profiles"]
    customer: dict = {}
    if customer_id:
        raw_customer = await asyncio.to_thread(
            db["Customer_Profiles"].find_one, {"customer_id": customer_id}
        )
        if raw_customer:
            customer = {k: v for k, v in raw_customer.items() if k != "_id"}
        else:
            logger.warning("No Customer_Profiles record for customer_id=%s", customer_id)

    # ── Claim history ──────────────────────────────────────────────────────────
    # Previously missing entirely — the MAS fraud agent fetches this from
    # Claim_History to detect frequent claimants and aggregate breaches.
    history_records: list[dict] = []
    if customer_id:
        try:
            raw_history = await asyncio.to_thread(
                lambda: list(
                    db["Claim_History"].find(
                        {"customer_id": customer_id}, {"_id": 0}
                    )
                )
            )
            history_records = raw_history
        except Exception as exc:
            logger.warning("Claim_History lookup failed for customer %s: %s", customer_id, exc)

    history_summary = _summarize_history(history_records)

    # ── Policy via Pinecone RAG ────────────────────────────────────────────────
    policy_meta: dict = {}
    if narrative and policy_id:
        try:
            vec = await asyncio.to_thread(get_embedder().encode, narrative)
            idx = get_pinecone_index()

            res = await asyncio.to_thread(
                idx.query,
                vector=vec.tolist(),
                top_k=1,
                filter={"policy_id": {"$eq": policy_id}},
                include_metadata=True,
            )

            if res.get("matches"):
                policy_meta = res["matches"][0]["metadata"]
            else:
                # Fallback: unfiltered query (same pattern as policy_agent.py)
                logger.warning(
                    "Exact policy filter returned no results for %s. Trying unfiltered.",
                    policy_id,
                )
                res_fallback = await asyncio.to_thread(
                    idx.query,
                    vector=vec.tolist(),
                    top_k=1,
                    include_metadata=True,
                )
                if res_fallback.get("matches"):
                    candidate = res_fallback["matches"][0]["metadata"]
                    if candidate.get("policy_id") == policy_id:
                        policy_meta = candidate
                    else:
                        logger.error(
                            "Fallback policy mismatch: got %s, expected %s",
                            candidate.get("policy_id"), policy_id,
                        )

        except Exception as exc:
            logger.error("Pinecone policy lookup failed: %s", exc)

    if not policy_meta:
        logger.warning(
            "No policy metadata retrieved for claim=%s policy=%s", claim_id, policy_id
        )

    logger.info(
        "fetch_nma_context | claim=%s | history_records=%d | policy_found=%s",
        claim_id, len(history_records), bool(policy_meta),
    )

    return {
        "claim": claim,
        "customer": customer,
        "history_summary": history_summary,  # new — plain text for prompt injection
        "history_records": history_records,  # new — raw list for deterministic checks
        "policy": policy_meta,
    }