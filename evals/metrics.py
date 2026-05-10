"""
metrics.py
----------
Pure-Python metric calculations.  All functions take raw result dicts
and ground-truth objects — no side effects, no I/O, easily unit-testable.

v2 fixes
--------
- `_check_decision`: `denial_reason_provided` step was always marked as
  executed because the old condition was:

      if fd.get("denial_reason") or fd.get("approved"):

  `fd.get("approved")` is True for every approved claim, so this check
  always passed — making step completeness artificially inflated for the
  decision_agent and masking real gaps in denial reasoning.

  Corrected logic: "denial_reason_provided" counts as executed if EITHER
    (a) the claim was denied AND a non-empty denial_reason is present, OR
    (b) the claim was approved (no denial reason is expected/required).
  A denied claim with an empty denial_reason is now correctly counted as
  an unexecuted step.

Metrics implemented
-------------------
1.  decision_accuracy          — fraction of correct Approve/Deny decisions
2.  fraud_precision_recall_f1  — standard IR metrics on fraud detection
3.  stp_rate                   — Straight-Through Processing rate
4.  consistency_score          — stability across K independent runs
5.  step_completeness          — executed vs. required reasoning steps
6.  latency_stats              — mean / p95 / total per-claim latency
7.  token_efficiency           — mean tokens per node per claim
"""

from __future__ import annotations

import math
import statistics
from typing import Any

from evals.ground_truth import GTRecord


# ══════════════════════════════════════════════════════════════════════════════
# 1. Decision Accuracy
# ══════════════════════════════════════════════════════════════════════════════

def decision_accuracy(
    results: list[dict],
    ground_truth: dict[str, GTRecord],
) -> dict[str, Any]:
    """
    Compare `final_decision.approved` against `ground_truth.expected_decision`.
    """
    correct = 0
    total = 0
    errors = []

    for r in results:
        cid = r.get("claim_id")
        gt = ground_truth.get(cid)
        if gt is None:
            continue

        fd = r.get("final_decision") or {}
        predicted = "Approved" if fd.get("approved") else "Denied"
        expected = gt.expected_decision

        total += 1
        if predicted == expected:
            correct += 1
        else:
            errors.append({
                "claim_id": cid,
                "predicted": predicted,
                "expected": expected,
                "fraud_scenario": gt.fraud_scenario,
            })

    return {
        "accuracy": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
        "misclassified": errors,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 2. Fraud Precision / Recall / F1
# ══════════════════════════════════════════════════════════════════════════════

def fraud_precision_recall_f1(
    results: list[dict],
    ground_truth: dict[str, GTRecord],
    risk_threshold: str = "High",
) -> dict[str, Any]:
    """
    Binary classification: fraud_report.risk_score == risk_threshold → predicted positive.
    """
    tp = fp = fn = tn = 0

    for r in results:
        cid = r.get("claim_id")
        gt = ground_truth.get(cid)
        if gt is None:
            continue

        fraud_report = r.get("fraud_report") or {}
        risk_score = (fraud_report.get("risk_score") or "").strip()

        predicted_positive = risk_score == risk_threshold
        actual_positive = gt.is_fraud

        if predicted_positive and actual_positive:
            tp += 1
        elif predicted_positive and not actual_positive:
            fp += 1
        elif not predicted_positive and actual_positive:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "threshold_used": risk_threshold,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 3. Straight-Through Processing Rate
# ══════════════════════════════════════════════════════════════════════════════

def stp_rate(results: list[dict]) -> dict[str, Any]:
    """
    A claim is straight-through if the failure_node was NOT reached.
    """
    FAILURE_MARKER = "Claim failed verification checks."

    straight_through = 0
    failed_early = 0
    total = len(results)

    for r in results:
        fd = r.get("final_decision") or {}
        denial = fd.get("denial_reason", "")

        hit_failure_node = FAILURE_MARKER in denial
        if not hit_failure_node:
            straight_through += 1
        else:
            failed_early += 1

    return {
        "stp_rate": round(straight_through / total, 4) if total else 0.0,
        "straight_through": straight_through,
        "failed_early": failed_early,
        "total": total,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4. Consistency Score
# ══════════════════════════════════════════════════════════════════════════════

def consistency_score(
    multi_run_results: dict[str, list[dict]],
) -> dict[str, Any]:
    """
    Given K runs per claim, compute how often the same final decision is produced.
    """
    per_claim: dict[str, float] = {}
    k_values = set()

    for cid, runs in multi_run_results.items():
        k = len(runs)
        k_values.add(k)
        if k < 2:
            per_claim[cid] = 1.0
            continue

        decisions = []
        for r in runs:
            fd = r.get("final_decision") or {}
            decisions.append(fd.get("approved"))

        majority = max(set(decisions), key=decisions.count)
        differing = sum(1 for d in decisions if d != majority)
        per_claim[cid] = round(1.0 - differing / k, 4)

    mean = statistics.mean(per_claim.values()) if per_claim else 0.0
    k_runs = max(k_values) if k_values else 0

    return {
        "mean_consistency": round(mean, 4),
        "std_consistency": (
            round(statistics.stdev(per_claim.values()), 4)
            if len(per_claim) > 1
            else 0.0
        ),
        "per_claim": per_claim,
        "k_runs": k_runs,
        "fully_consistent_claims": sum(1 for v in per_claim.values() if v == 1.0),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 5. Step Completeness
# ══════════════════════════════════════════════════════════════════════════════

REQUIRED_STEPS: dict[str, list[str]] = {
    "verification_agent": [
        "claim_fetched",
        "ocr_parsed",
        "name_check_performed",
        "policy_check_performed",
    ],
    "policy_agent": [
        "policy_retrieved",
        "remaining_limit_computed",
        "exclusion_checked",
        "coverage_determined",
        "coverage_reasoning",
    ],
    "fraud_agent": [
        "history_fetched",
        "frequent_claims_checked",
        "collusion_checked",
        "staging_checked",
        "risk_score_assigned",
        "reasoning_provided",
    ],
    "decision_agent": [
        "payout_calculated",
        "decision_made",
        "denial_reason_provided",
        "step_by_step_reasoning",
    ],
}


def _check_verification(state: dict) -> list[str]:
    executed = []
    if state.get("raw_data"):
        executed.append("claim_fetched")
    vo = state.get("verification_output") or {}
    if isinstance(vo, dict):
        if vo.get("ocr"):
            executed.append("ocr_parsed")
        if "name_match" in vo:
            executed.append("name_check_performed")
        if "policy_match" in vo:
            executed.append("policy_check_performed")
    return executed


def _check_policy(state: dict) -> list[str]:
    executed = []
    pv = state.get("policy_verdict") or {}
    if not isinstance(pv, dict):
        return executed
    if pv.get("policy_id"):
        executed.append("policy_retrieved")
    if "remaining_limit" in pv:
        executed.append("remaining_limit_computed")
    if "exclusion_triggered" in pv:
        executed.append("exclusion_checked")
    if "incident_covered" in pv:
        executed.append("coverage_determined")
    if pv.get("coverage_reasoning"):
        executed.append("coverage_reasoning")
    return executed


def _check_fraud(state: dict) -> list[str]:
    executed = []
    fr = state.get("fraud_report") or {}
    if not isinstance(fr, dict):
        return executed
    if fr:
        executed.append("history_fetched")
    if "frequent_claims_flag" in fr:
        executed.append("frequent_claims_checked")
    if "collusion_flag" in fr:
        executed.append("collusion_checked")
    if "staging_flag" in fr:
        executed.append("staging_checked")
    if fr.get("risk_score"):
        executed.append("risk_score_assigned")
    if fr.get("reasoning"):
        executed.append("reasoning_provided")
    return executed


def _check_decision(state: dict) -> list[str]:
    """
    BUG FIX: `denial_reason_provided` was always True because the old code
    used `fd.get("approved")` as a fallback condition, which is True for
    every approved claim — making the step appear executed even when no
    denial_reason was produced.

    Correct semantics:
      - If the claim was DENIED → the step is executed only if denial_reason
        is present and non-empty.
      - If the claim was APPROVED → no denial_reason is needed; the step is
        considered satisfied (there is nothing to check).
    """
    executed = []
    fd = state.get("final_decision") or {}
    if not isinstance(fd, dict):
        return executed

    if state.get("final_payout") is not None:
        executed.append("payout_calculated")

    if "approved" in fd:
        executed.append("decision_made")

    # Fixed: approved=True means no denial reason is expected → step satisfied.
    # approved=False (denied) → step satisfied only if denial_reason is non-empty.
    approved = fd.get("approved")
    denial_reason = fd.get("denial_reason", "")
    if approved is True or (approved is False and bool(denial_reason)):
        executed.append("denial_reason_provided")

    if fd.get("step_by_step_reasoning"):
        executed.append("step_by_step_reasoning")

    return executed


def step_completeness(results: list[dict]) -> dict[str, Any]:
    """
    For each agent node, compute what fraction of required steps were executed.
    """
    _checkers = {
        "verification_agent": _check_verification,
        "policy_agent": _check_policy,
        "fraud_agent": _check_fraud,
        "decision_agent": _check_decision,
    }

    per_claim: dict[str, dict[str, float]] = {}
    agent_totals: dict[str, list[float]] = {a: [] for a in _checkers}

    for r in results:
        cid = r.get("claim_id", "unknown")
        per_claim[cid] = {}
        for agent, checker in _checkers.items():
            executed = checker(r)
            required = REQUIRED_STEPS[agent]
            ratio = len(set(executed) & set(required)) / len(required)
            per_claim[cid][agent] = round(ratio, 4)
            agent_totals[agent].append(ratio)

    per_agent = {
        agent: {
            "mean": round(statistics.mean(vals), 4) if vals else 0.0,
            "required_steps": len(REQUIRED_STEPS[agent]),
        }
        for agent, vals in agent_totals.items()
    }

    all_vals = [v for claim in per_claim.values() for v in claim.values()]
    mean_completeness = round(statistics.mean(all_vals), 4) if all_vals else 0.0

    return {
        "mean_completeness": mean_completeness,
        "per_agent": per_agent,
        "per_claim": per_claim,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 6. Latency Statistics
# ══════════════════════════════════════════════════════════════════════════════

def latency_stats(timing_records: list[dict]) -> dict[str, Any]:
    """
    Parameters
    ----------
    timing_records : [{"claim_id": str, "total_s": float, "per_agent": {...}}]
    """
    totals = [t["total_s"] for t in timing_records if "total_s" in t]
    if not totals:
        return {}

    sorted_totals = sorted(totals)
    p95_idx = max(0, math.ceil(0.95 * len(sorted_totals)) - 1)

    agent_times: dict[str, list[float]] = {}
    for t in timing_records:
        for agent, elapsed in (t.get("per_agent") or {}).items():
            agent_times.setdefault(agent, []).append(elapsed)

    return {
        "mean_latency_s": round(statistics.mean(totals), 3),
        "median_latency_s": round(statistics.median(totals), 3),
        "p95_latency_s": round(sorted_totals[p95_idx], 3),
        "max_latency_s": round(max(totals), 3),
        "total_wall_s": round(sum(totals), 3),
        "per_agent_mean_s": {
            agent: round(statistics.mean(vals), 3)
            for agent, vals in agent_times.items()
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# 7. Token Efficiency
# ══════════════════════════════════════════════════════════════════════════════

def token_efficiency(token_records: list[dict]) -> dict[str, Any]:
    """
    Parameters
    ----------
    token_records : [{"claim_id": str, "per_agent": {"agent_name": int}}]
    """
    per_agent: dict[str, list[int]] = {}
    totals_per_claim: list[int] = []

    for record in token_records:
        claim_total = 0
        for agent, tokens in (record.get("per_agent") or {}).items():
            per_agent.setdefault(agent, []).append(tokens)
            claim_total += tokens
        totals_per_claim.append(claim_total)

    return {
        "mean_total_tokens_per_claim": (
            round(statistics.mean(totals_per_claim), 1) if totals_per_claim else 0.0
        ),
        "per_agent_mean_tokens": {
            agent: round(statistics.mean(vals), 1)
            for agent, vals in per_agent.items()
        },
        "total_tokens_all_claims": sum(totals_per_claim),
    }