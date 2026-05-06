"""
main.py
-------
Entry point for the Car Insurance MAS.

Startup pre-warming
-------------------
The SentenceTransformer model (all-MiniLM-L6-v2) takes ~3-10s to load on first
use. Without pre-warming, the FIRST claim processed by a fresh process pays this
penalty. Pre-warming loads the model at import time so every request after that
hits the LRU cache immediately.

The Groq key rotation manager is also initialized at import time for the same reason.
"""

import sys
import json
import logging

logger = logging.getLogger(__name__)


# ── Pre-warm singletons at import time ────────────────────────────────────────
# This runs once when the module is first imported (app startup).
# Subsequent calls hit the lru_cache and return instantly.

def _prewarm():
    try:
        from src.agents.policy_agent import _get_embedder
        logger.info("Pre-warming SentenceTransformer model…")
        _get_embedder()
        logger.info("SentenceTransformer model ready.")
    except Exception as exc:
        logger.warning("SentenceTransformer pre-warm failed (non-fatal): %s", exc)

    try:
        from src.tools.groq_client import _get_manager
        logger.info("Initializing Groq key rotation manager…")
        _get_manager()
        logger.info("Groq client ready.")
    except Exception as exc:
        logger.warning("Groq client pre-warm failed (non-fatal): %s", exc)


_prewarm()

# ── Import graph after pre-warming ────────────────────────────────────────────
from src.graph.graph import compiled_graph
from src.graph.state import ClaimState


def process_claim(claim_id: str) -> ClaimState:
    initial_state: ClaimState = {
        "claim_id": claim_id,
        "customer_profile": {},
        "sanitized_data": None,
        "policy_verdict": None,
        "fraud_report": None,
        "final_decision": None,
        "final_payout": None,
        "errors": [],
    }
    return compiled_graph.invoke(initial_state)


def _print_section(title: str, data) -> None:
    print(f"\n{'─' * 50}")
    print(f"  {title}")
    print(f"{'─' * 50}")
    if isinstance(data, dict):
        print(json.dumps(data, indent=2, default=str))
    else:
        print(data)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    claim_id = sys.argv[1] if len(sys.argv) > 1 else ""
    if not claim_id:
        print("Usage: python -m src.main <claim_id>")
        sys.exit(1)

    print(f"\n{'═' * 50}")
    print(f"  Processing Claim: {claim_id}")
    print(f"{'═' * 50}")

    result = process_claim(claim_id)
    decision = result.get("final_decision") or {}

    _print_section("SANITIZED DATA", result.get("sanitized_data"))
    _print_section("POLICY VERDICT", result.get("policy_verdict"))
    _print_section("FRAUD REPORT", result.get("fraud_report"))
    _print_section("ERRORS", result.get("errors") or "None")

    print(f"\n{'═' * 50}")
    status = " APPROVED" if decision.get("approved") else " DENIED"
    print(f"  FINAL DECISION: {status}")
    print(f"  FINAL PAYOUT  : ${result.get('final_payout', 0.0):,.2f}")
    print(f"{'═' * 50}")
    print(f"\nReasoning:\n{decision.get('step_by_step_reasoning', 'N/A')}")