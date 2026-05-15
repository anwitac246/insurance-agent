"""
run_evals_nma.py
----------------
Evaluation harness for the NMA (single-agent) system.

Speed improvements in this version
------------------------------------
- INTER_CLAIM_DELAY reduced from 3.0s → 0.0s (Ollama is local, no rate limits).
- Claims are now processed in concurrent batches (CONCURRENCY=5) using
  asyncio.gather(), matching the MAS eval runner's approach.
  Each NMA claim fires 1 LLM call vs the MAS's 3, so NMA concurrency headroom
  is higher — set CONCURRENCY=8 if your GPU has headroom.

Accuracy fix
------------
- _compute_signals(): collusion shop match now uses full-name substring
  (case-insensitive) instead of first-word-only ("apex"), matching the fix
  applied to fraud_agent.py.
"""

import sys
import os
import asyncio
import time
import logging
from collections import Counter, defaultdict
from datetime import datetime

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
nma_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
if nma_dir not in sys.path:
    sys.path.insert(0, nma_dir)

from dotenv import load_dotenv
load_dotenv()

from nma_src.tools.context_fetcher import fetch_nma_context
from nma_src.agents.nma_agent import arun_nma_agent
from src.tools.mongo_client import get_db

from evals.reporter import save_json_report, build_report_chart
from evals.ground_truth import load_ground_truth, GTRecord
from evals.metrics import (
    decision_accuracy,
    fraud_precision_recall_f1,
    stp_rate,
    latency_stats,
)
from evals.run_evals import _run_llm_judge

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# FIX: 0s delay — Ollama is local, no rate limits to respect
INTER_BATCH_DELAY = 0.0
# Concurrent NMA claims per batch. NMA fires 1 LLM call/claim (vs MAS's 3),
# so can tolerate higher concurrency.
CONCURRENCY = 8


def _make_failure_result(claim_id: str, error: str) -> dict:
    return {
        "claim_id": claim_id,
        "latency_seconds": 0.0,
        "final_decision": {
            "approved": False,
            "final_payout": 0.0,
            "step_by_step_reasoning": f"Claim processing failed: {error}",
        },
        "fraud_report": {
            "risk_score": "Low",
            "anomalies": [],
        },
        "policy_verdict": {
            "incident_covered": False,
            "exclusion_triggered": False,
            "exclusion_reason": "",
        },
        "errors": [error],
    }


def _scenario_breakdown(results: list[dict], ground_truth: dict[str, GTRecord]) -> dict:
    stats: dict[str, dict] = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        cid = r.get("claim_id")
        gt = ground_truth.get(cid)
        if not gt:
            continue
        fd = r.get("final_decision") or {}
        predicted = "Approved" if fd.get("approved") else "Denied"
        scenario = gt.fraud_scenario
        stats[scenario]["total"] += 1
        if predicted == gt.expected_decision:
            stats[scenario]["correct"] += 1
    return {
        scenario: {
            "accuracy": round(v["correct"] / v["total"], 4) if v["total"] else 0.0,
            "correct": v["correct"],
            "total": v["total"],
        }
        for scenario, v in stats.items()
    }


def _print_summary(report: dict) -> None:
    sep = "─" * 55
    print(f"\n{'═' * 55}")
    print(f"  NMA EVALUATION COMPLETE  [{report['run_id']}]")
    print(f"{'═' * 55}")
    print(f"  Claims evaluated     : {report['claims_evaluated']}")
    wall = report.get("total_time_seconds", 0)
    if wall:
        per_claim = wall / max(report["claims_evaluated"], 1)
        print(f"  Wall time            : {wall:.0f}s  ({per_claim:.1f}s/claim)")
    print(sep)
    acc = report["decision_accuracy"]
    print(f"  Decision Accuracy    : {acc['accuracy']*100:.1f}%  ({acc['correct']}/{acc['total']})")
    fm = report["fraud_metrics"]
    print(f"  Fraud Precision      : {fm['precision']*100:.1f}%")
    print(f"  Fraud Recall         : {fm['recall']*100:.1f}%")
    print(f"  Fraud F1             : {fm['f1']*100:.1f}%  "
          f"(TP={fm['tp']} FP={fm['fp']} FN={fm['fn']})")
    stp = report["stp"]
    print(f"  STP Rate             : {stp['stp_rate']*100:.1f}%  "
          f"({stp['straight_through']}/{stp['total']})")
    lat = report["latency"]
    if lat:
        print(f"  Mean Latency         : {lat.get('mean_latency_s', 0):.2f}s  "
              f"(p95={lat.get('p95_latency_s', 0):.2f}s)")
    print(f"{'═' * 55}\n")
    print("  Per-Scenario Accuracy:")
    for scenario, stats in sorted(report.get("scenario_breakdown", {}).items()):
        bar_len = int(stats["accuracy"] * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(
            f"    {scenario:<22} {bar}  "
            f"{stats['accuracy']*100:.0f}%  ({stats['correct']}/{stats['total']})"
        )
    print()


async def _process_single_claim(claim_id: str) -> tuple[dict, float]:
    """Process one NMA claim and return (result_dict, elapsed_seconds)."""
    t0 = time.perf_counter()
    try:
        ctx = await fetch_nma_context(claim_id)
        res = await arun_nma_agent(ctx)
        lat = time.perf_counter() - t0

        result = {
            "claim_id": claim_id,
            "latency_seconds": lat,
            "final_decision": {
                "approved": res.approved,
                "final_payout": res.final_payout,
                "step_by_step_reasoning": res.step_by_step_reasoning,
            },
            "fraud_report": {
                "risk_score": (
                    "High" if res.fraud_risk_score >= 8
                    else "Medium" if res.fraud_risk_score >= 4
                    else "Low"
                ),
                "anomalies": (
                    [res.fraud_anomalies] if res.fraud_anomalies != "None" else []
                ),
            },
            "policy_verdict": {
                "incident_covered": res.incident_covered,
                "exclusion_triggered": res.exclusion_triggered,
                "exclusion_reason": res.exclusion_reason,
                "coverage_scope": ctx.get("policy", {}).get("coverage_scope", ""),
                "exclusions": ctx.get("policy", {}).get("exclusions", ""),
                "policy_limit": ctx.get("policy", {}).get("policy_limit", 0),
                "aggregate_limit": ctx.get("policy", {}).get("aggregate_limit", 0),
                "deductible": ctx.get("policy", {}).get("deductible", 0),
                "total_historical_payout": ctx.get("policy", {}).get("total_historical_payout", 0),
                "remaining_limit": (
                    float(ctx.get("policy", {}).get("aggregate_limit", 0))
                    - float(ctx.get("policy", {}).get("total_historical_payout", 0))
                ),
            },
            "errors": [],
        }
        logger.info(
            "Claim %s → approved=%s | payout=$%.2f | fraud_score=%d | %.2fs",
            claim_id, res.approved, res.final_payout, res.fraud_risk_score, lat,
        )
        return result, lat

    except Exception as exc:
        lat = time.perf_counter() - t0
        logger.error("Failed claim %s after %.2fs: %s", claim_id, lat, exc)
        result = _make_failure_result(claim_id, str(exc))
        result["latency_seconds"] = lat
        return result, lat


async def _process_batch(claim_ids: list[str], batch_num: int, total_batches: int) -> list[tuple[dict, float]]:
    logger.info(
        "Batch [%d/%d]: processing %d claims concurrently…",
        batch_num, total_batches, len(claim_ids),
    )
    tasks = [_process_single_claim(cid) for cid in claim_ids]
    raw = await asyncio.gather(*tasks, return_exceptions=True)

    processed = []
    for cid, res in zip(claim_ids, raw):
        if isinstance(res, Exception):
            logger.error("Claim %s batch task raised: %s", cid, res)
            processed.append((_make_failure_result(cid, str(res)), 0.0))
        else:
            processed.append(res)
    return processed


async def run_evals_nma():
    db = get_db()
    claims_cursor = db["Active_Claims"].find({}, {"_id": 0, "claim_id": 1})
    claim_ids: list[str] = await asyncio.to_thread(
        lambda: [c["claim_id"] for c in claims_cursor]
    )

    if not claim_ids:
        logger.error("No claims found in Active_Claims. Run scripts/seed_data.py first.")
        return

    ground_truth = load_ground_truth()

    results: list[dict] = []
    latencies: list[dict] = []

    # Split into batches
    batches = [
        claim_ids[i:i + CONCURRENCY]
        for i in range(0, len(claim_ids), CONCURRENCY)
    ]
    total_batches = len(batches)

    start_run = time.perf_counter()

    for batch_idx, batch in enumerate(batches, 1):
        batch_results = await _process_batch(batch, batch_idx, total_batches)

        for cid, (result, lat) in zip(batch, batch_results):
            results.append(result)
            latencies.append({"claim_id": cid, "total_s": lat})

        if INTER_BATCH_DELAY > 0 and batch_idx < total_batches:
            await asyncio.sleep(INTER_BATCH_DELAY)

    total_time = time.perf_counter() - start_run
    logger.info(
        "All %d claims processed in %.1fs (%.1fs/claim average, concurrency=%d)",
        len(claim_ids), total_time, total_time / len(claim_ids), CONCURRENCY,
    )

    # ── Metrics ────────────────────────────────────────────────────────────────
    acc_metrics = decision_accuracy(results, ground_truth)
    fraud_metrics = fraud_precision_recall_f1(results, ground_truth)
    stp_metrics = stp_rate(results)
    lat_metrics = latency_stats(latencies)
    breakdown = _scenario_breakdown(results, ground_truth)

    judge_metrics = _run_llm_judge(results, ground_truth, max_sample=5)

    run_id = f"NMA_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    report = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "system": "NMA (Single Agent)",
        "claims_evaluated": len(results),
        "concurrency": CONCURRENCY,
        "total_time_seconds": round(total_time, 2),
        "decision_accuracy": acc_metrics,
        "fraud_metrics": fraud_metrics,
        "stp": stp_metrics,
        "latency": lat_metrics,
        "token_efficiency": {},
        "scenario_breakdown": breakdown,
        "consistency": {"mean_consistency": 1.0, "per_claim": {}, "k_runs": 1},
        "step_completeness": {},
        "llm_judge": judge_metrics,
        "chaos": {},
    }

    out_dir = os.path.join(root_dir, "evals", "results")
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, f"eval_report_{run_id}.json")
    save_json_report(report, json_path)

    _print_summary(report)
    logger.info("Report saved to %s", json_path)
    return report


if __name__ == "__main__":
    asyncio.run(run_evals_nma())