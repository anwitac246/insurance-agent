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

The parallel_analysis node uses asyncio.gather to fire both the policy and
fraud agents simultaneously, then merges their outputs before the decision
agent runs. On a single claim this saves ~4–8s (one full 70B LLM round-trip).
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


# ── Routing ────────────────────────────────────────────────────────────────────

def should_continue(state: ClaimState) -> str:
    return "failure_node" if state.get("errors") else "parallel_analysis"


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

        merged = dict(state)  # start from current state

        # ── Merge policy result ────────────────────────────────────────────────
        if isinstance(policy_result, Exception):
            logger.error("policy_agent raised an exception: %s", policy_result)
            merged["policy_verdict"] = None
            merged["errors"] = list(merged.get("errors", [])) + [
                f"Policy agent exception: {policy_result}"
            ]
        else:
            merged["policy_verdict"] = policy_result.get("policy_verdict")
            # Merge any new errors the policy agent appended
            merged["errors"] = list(policy_result.get("errors", []))

        # ── Merge fraud result ─────────────────────────────────────────────────
        if isinstance(fraud_result, Exception):
            logger.error("fraud_agent raised an exception: %s", fraud_result)
            merged["fraud_report"] = None
            merged["errors"] = list(merged.get("errors", [])) + [
                f"Fraud agent exception: {fraud_result}"
            ]
        else:
            merged["fraud_report"] = fraud_result.get("fraud_report")
            # Union the error lists from both agents (avoid duplicates)
            existing_errors = set(merged.get("errors", []))
            for err in fraud_result.get("errors", []):
                if err not in existing_errors:
                    merged.setdefault("errors", [])
                    merged["errors"].append(err)
                    existing_errors.add(err)

        return merged

    # LangGraph node functions are sync; bridge back to sync here.
    # If there is already a running event loop (e.g. inside FastAPI), use
    # asyncio.ensure_future + loop.run_until_complete is not safe — instead we
    # create a new loop explicitly so this is always safe regardless of caller.
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # We are inside an async context (FastAPI, pytest-asyncio, etc.)
            # Schedule the coroutine as a task and block until done.
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, _run())
                return future.result()
        else:
            return loop.run_until_complete(_run())
    except RuntimeError:
        # No event loop at all — create one
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