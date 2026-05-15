"""
run_evals_nma.py
----------------
Evaluation harness for the NMA (single-agent) system.
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
from src.tools.llm_client import adaptive_sleep

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

# Seconds to sleep between claims — each NMA claim fires 1 LLM call
# (half the MAS rate, which fires 2 in parallel), so 3s is safe on free tier.
INTER_CLAIM_DELAY = 3.0


def _make_failure_result(claim_id: str, error: str) -> dict:
    """
    Fallback result when a claim fails completely.
    Mirrors the failure format used by run_evals.py so metrics functions
    receive a consistent structure.
    """
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


async def run_evals_nma():
    db = get_db()

    # BUG FIX: use explicit dict-style collection access for clarity
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
    start_run = time.perf_counter()

    for i, cid in enumerate(claim_ids, 1):
        logger.info("Processing [%d/%d] %s", i, len(claim_ids), cid)
        t0 = time.perf_counter()

        try:
            ctx = await fetch_nma_context(cid)
            res = await arun_nma_agent(ctx)
            lat = time.perf_counter() - t0

            result = {
                "claim_id": cid,
                "latency_seconds": lat,
                "final_decision": {
                    "approved": res.approved,
                    "final_payout": res.final_payout,
                    "step_by_step_reasoning": res.step_by_step_reasoning,
                },
                "fraud_report": {
                    # Map numeric score to string risk level used by fraud metrics
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
                    "remaining_limit": float(ctx.get("policy", {}).get("aggregate_limit", 0)) - float(ctx.get("policy", {}).get("total_historical_payout", 0)),
                },
                "errors": [],
            }
            results.append(result)
            logger.info(
                "Claim %s → approved=%s | payout=$%.2f | fraud_score=%d | %.2fs",
                cid, res.approved, res.final_payout, res.fraud_risk_score, lat,
            )

        except Exception as exc:
            lat = time.perf_counter() - t0
            logger.error("Failed claim %s after %.2fs: %s", cid, lat, exc)
            # BUG FIX: previously this path silently dropped the claim from results.
            # Always append a failure record so the eval denominator is correct.
            result = _make_failure_result(cid, str(exc))
            result["latency_seconds"] = lat
            results.append(result)

        latencies.append({"claim_id": cid, "total_s": time.perf_counter() - t0})

        # Rate-limit backoff between claims
        if i < len(claim_ids):
            actual = adaptive_sleep(INTER_CLAIM_DELAY)
            logger.debug("Slept %.1fs before next claim", actual)

    total_time = time.perf_counter() - start_run

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
        "decision_accuracy": acc_metrics,
        "fraud_metrics": fraud_metrics,
        "stp": stp_metrics,
        "latency": lat_metrics,
        "token_efficiency": {},
        "total_time_seconds": round(total_time, 2),
        "scenario_breakdown": breakdown,
        # These keys keep the reporter happy when building comparison charts
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