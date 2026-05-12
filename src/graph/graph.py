"""
graph.py
--------
LangGraph pipeline with parallel policy + fraud execution.

Architecture
------------
                    ┌─────────────────────┐
                    │  document_verification│
                    └──────────┬──────────┘
                               │
              ┌────────────────┴────────────────┐
              ▼                                  ▼
        failure_node                    parallel_analysis
              │                   (policy_agent ∥ fraud_agent)
              │                                  │
              └──────────────┬───────────────────┘
                             ▼
                          decision
                             │
                            END

BUG FIXES vs previous version
------------------------------
1. should_continue() now routes to failure_node ONLY for FATAL errors
   (claim not found / customer profile missing).  Non-fatal discrepancies
   like OCR name-mismatch or policy-number-mismatch are demoted to
   `warnings` and carried forward into parallel_analysis so the fraud and
   policy agents can still evaluate the claim.  Previously ANY error —
   including a trivial name mismatch — short-circuited the entire pipeline
   and hard-denied the claim, which was the single largest driver of false
   denials and lower MAS accuracy vs NMA.

2. parallel_analysis error merge is now a true UNION of:
     verification errors  +  policy errors  +  fraud errors
   Previously `merged["errors"] = list(policy_result.get("errors", []))`
   *replaced* all errors (including verification) with just the policy
   agent's list, and fraud errors were added back inconsistently.

3. The `warnings` key is added to ClaimState to hold non-fatal verification
   discrepancies (name/policy-number mismatch) without triggering routing to
   the failure node.

4. FIX: Fatal prefix matching now uses substring containment instead of
   startswith("Claim "). The old prefix "Claim " matched error messages like
   "Claim exclusion triggered: Street Racing" from policy_agent.py, silently
   routing valid-but-excluded claims to failure_node instead of decision.
   This was causing semantic_exclusion claims to be hard-denied at the wrong
   stage and corrupting per-scenario accuracy metrics.
"""

import asyncio
import logging

from langgraph.graph import StateGraph, END

from src.graph.state import ClaimState
from src.agents.verification_agent import arun_verification_agent
from src.agents.policy_agent import arun_policy_agent
from src.agents.fraud_agent import arun_fraud_agent
from src.agents.decision_agent import arun_decision_agent

logger = logging.getLogger(__name__)

# ── Fatal error markers ────────────────────────────────────────────────────────
# FIX: Use substring containment instead of startswith().
# The old prefix "Claim " matched policy_agent errors like:
#   "Claim exclusion triggered: Street Racing"
# which caused valid excluded claims to be routed to failure_node instead
# of reaching the decision agent where the exclusion would be applied correctly.
#
# Only errors that originate from verification_agent for missing DB records
# are truly fatal — everything else is a denial signal for the decision agent.
_FATAL_ERROR_SUBSTRINGS = (
    "not found in Active_Claims",   # from verification_agent when claim is missing
    "No customer profile found",    # from verification_agent when customer is missing
)


def _is_fatal(error: str) -> bool:
    """
    Returns True only for errors that indicate a missing DB record.
    All other errors (fraud signals, policy exclusions, aggregate breaches)
    are denial signals that must reach the decision agent, not fatal routing signals.
    """
    return any(s in error for s in _FATAL_ERROR_SUBSTRINGS)


# ── Routing ────────────────────────────────────────────────────────────────────

def should_continue(state: ClaimState) -> str:
    """
    Route to failure_node only when a FATAL error is present (claim or customer
    not found).  Non-fatal discrepancies (OCR name mismatch, policy number
    mismatch) are warnings — the claim proceeds to parallel_analysis.

    Previously ANY non-empty errors list triggered failure_node, which caused
    normal claims with minor OCR discrepancies to be hard-denied before the
    policy and fraud agents could evaluate them.
    """
    fatal = [e for e in state.get("errors", []) if _is_fatal(e)]
    if fatal:
        logger.warning(
            "claim=%s | FATAL verification error(s) → failure_node: %s",
            state.get("claim_id"), fatal,
        )
        return "failure_node"
    return "parallel_analysis"


# ── Failure short-circuit ──────────────────────────────────────────────────────

def failure_node(state: ClaimState) -> ClaimState:
    return {
        **state,
        "final_payout": 0.0,
        "final_decision": {
            "approved": False,
            "denial_reason": "Claim failed verification checks.",
            "errors": state["errors"],
            "step_by_step_reasoning": (
                "Claim was rejected at the Document Verification stage. "
                f"Errors: {'; '.join(state['errors'])}"
            ),
        },
    }


# ── Parallel analysis node ─────────────────────────────────────────────────────

def parallel_analysis_node(state: ClaimState) -> ClaimState:
    """
    Runs the policy agent and fraud agent concurrently using asyncio.gather.

    Both agents only depend on `sanitized_data` from the verification step,
    so they are fully independent and safe to parallelise.

    BUG FIX: Error merging is now a true union across all three stages:
      verification errors  ∪  policy errors  ∪  fraud errors
    Previously the policy errors *replaced* the verification error list,
    causing verification context to be silently dropped.

    Timing example (single claim, Groq free tier):
        Sequential:  policy(~5s) + fraud(~7s) = ~12s
        Parallel:    max(policy, fraud)        =  ~7s   ← ~40% faster
    """
    async def _run():
        policy_task = arun_policy_agent(state)
        fraud_task  = arun_fraud_agent(state)

        policy_result, fraud_result = await asyncio.gather(
            policy_task, fraud_task, return_exceptions=True
        )

        merged = dict(state)  # start from current state; preserves verification errors

        # ── True union error accumulator ───────────────────────────────────────
        # Start from the verification errors that are already in merged["errors"].
        # Each agent may add new errors; we union them all without duplicates.
        accumulated_errors: list[str] = list(merged.get("errors", []))
        seen_errors: set[str] = set(accumulated_errors)

        def _union_errors(new_errors: list[str]) -> None:
            for err in new_errors:
                if err not in seen_errors:
                    accumulated_errors.append(err)
                    seen_errors.add(err)

        # ── Merge policy result ────────────────────────────────────────────────
        if isinstance(policy_result, Exception):
            logger.error("policy_agent raised an exception: %s", policy_result)
            merged["policy_verdict"] = None
            _union_errors([f"Policy agent exception: {policy_result}"])
        else:
            merged["policy_verdict"] = policy_result.get("policy_verdict")
            _union_errors(policy_result.get("errors", []))

        # ── Merge fraud result ─────────────────────────────────────────────────
        if isinstance(fraud_result, Exception):
            logger.error("fraud_agent raised an exception: %s", fraud_result)
            merged["fraud_report"] = None
            _union_errors([f"Fraud agent exception: {fraud_result}"])
        else:
            merged["fraud_report"] = fraud_result.get("fraud_report")
            _union_errors(fraud_result.get("errors", []))

        merged["errors"] = accumulated_errors
        return merged

    # LangGraph node functions are sync; bridge back to sync here.
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, _run())
                return future.result()
        else:
            return loop.run_until_complete(_run())
    except RuntimeError:
        return asyncio.run(_run())


# ── Decision node ──────────────────────────────────────────────────────────────

def decision_node(state: ClaimState) -> ClaimState:
    """Sync wrapper around the async decision agent."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, arun_decision_agent(state))
                return future.result()
        else:
            return loop.run_until_complete(arun_decision_agent(state))
    except RuntimeError:
        return asyncio.run(arun_decision_agent(state))


# ── Verification node ──────────────────────────────────────────────────────────

def verification_node(state: ClaimState) -> ClaimState:
    """Sync wrapper around the async verification agent."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, arun_verification_agent(state))
                return future.result()
        else:
            return loop.run_until_complete(arun_verification_agent(state))
    except RuntimeError:
        return asyncio.run(arun_verification_agent(state))


# ── Graph assembly ─────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(ClaimState)

    graph.add_node("document_verification", verification_node)
    graph.add_node("failure_node", failure_node)
    graph.add_node("parallel_analysis", parallel_analysis_node)
    graph.add_node("decision", decision_node)

    graph.set_entry_point("document_verification")

    graph.add_conditional_edges(
        "document_verification",
        should_continue,
        {
            "failure_node": "failure_node",
            "parallel_analysis": "parallel_analysis",
        },
    )

    graph.add_edge("failure_node", END)
    graph.add_edge("parallel_analysis", "decision")
    graph.add_edge("decision", END)

    return graph


compiled_graph = build_graph().compile()