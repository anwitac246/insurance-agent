"""
run_evals.py
------------
Top-level evaluation script for the Car Insurance MAS.

Run
---
    python -m evals.run_evals [--k 3] [--no-chaos] [--no-llm-judge]
                              [--output-dir evals/results] [--claims N]
                              [--claim-ids id1 id2 ...] [--delay 4]

Arguments
---------
--k                  Number of repeated runs per claim for consistency testing (default 1).
                     Set to 3 for a full consistency sweep — note this multiplies runtime.
--no-chaos           Skip chaos/corruption sweep (faster).
--no-llm-judge       Skip Groundedness and Hallucination scoring (faster).
--output-dir         Directory for JSON report and PNG chart (default: evals/results).
--claims N           Process only the first N claims (default: all).
--claim-ids          Explicit list of claim IDs to process.
--delay              Seconds to sleep between claims (default: 4).
                     NOTE: The parallel_analysis node fires policy_agent AND fraud_agent
                     simultaneously, so each claim consumes ~2x the RPM budget. The
                     default of 4s gives enough headroom on Groq free tier (30 RPM).
                     Set to 0 to disable throttling entirely (risky on free tier).
--llm-judge-sample   Max claims to score with LLM judge (default: 5).

Exit codes
----------
0 — all metrics computed successfully
1 — fatal error (missing env vars, empty DB, etc.)

Speed notes
-----------
Default settings (--k 1 --no-chaos --llm-judge-sample 5) run ~50 claims sequentially
with a 4s inter-claim delay: ~50 × (LLM latency + 4s) ≈ 8–12 min.

To get a fast smoke-test result in ~2 min:
    python -m evals.run_evals --claims 10 --k 1 --no-chaos --no-llm-judge --delay 2

For a full production eval, use:
    python -m evals.run_evals --k 3 --llm-judge-sample 10
"""

from __future__ import annotations

import argparse
import logging
import os
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv()

from evals.ground_truth import load_ground_truth, GTRecord
from evals.metrics import (
    decision_accuracy,
    fraud_precision_recall_f1,
    stp_rate,
    consistency_score,
    step_completeness,
    latency_stats,
    token_efficiency,
)
from evals.chaos import ChaosHarness
from evals.reporter import build_report_chart, save_json_report
from src.tools.groq_client import adaptive_sleep

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("evals.run_evals")


# ── Import the MAS ─────────────────────────────────────────────────────────────
try:
    from src.main import process_claim as _process_claim
except ImportError as exc:
    logger.error(
        "Cannot import src.main.process_claim. "
        "Make sure you run this from the project root: %s", exc
    )
    sys.exit(1)


# ── Instrumented wrapper ───────────────────────────────────────────────────────

def _timed_process(claim_id: str) -> tuple[dict, float]:
    t0 = time.perf_counter()
    try:
        result = _process_claim(claim_id)
    except Exception as exc:
        logger.error("Pipeline raised an exception for claim %s: %s", claim_id, exc)
        result = {
            "claim_id": claim_id,
            "final_decision": {"approved": False, "denial_reason": str(exc)},
            "errors": [str(exc)],
            "final_payout": 0.0,
        }
    elapsed = time.perf_counter() - t0
    return result, elapsed


def _extract_token_usage(result: dict) -> dict[str, int]:
    usage = result.get("token_usage") or {}
    if not usage:
        return {
            "verification_agent": 0,
            "policy_agent": 0,
            "fraud_agent": 0,
            "decision_agent": 0,
        }
    return usage


# ── LLM-Judge scoring (optional) ──────────────────────────────────────────────

def _run_llm_judge(
    results: list[dict],
    ground_truth: dict[str, GTRecord],
    max_sample: int = 5,
) -> dict:
    """
    Run Groundedness and Hallucination judges on a sample of results.

    max_sample is intentionally small (default 5) to avoid rate-limit 429s
    and keep eval runtime reasonable. Raise it with --llm-judge-sample if needed.

    Both judges now return None on failure (400 / exhausted retries) rather than
    raising — the eval run continues and the failed claims are counted separately.
    """
    try:
        from evals.llm_judge import GroundednessJudge, HallucinationJudge
    except ImportError as exc:
        logger.warning("LLM judge import failed: %s", exc)
        return {"error": str(exc)}

    g_judge = GroundednessJudge()
    h_judge = HallucinationJudge()

    groundedness_scores: list[int] = []
    hallucination_rates: list[float] = []
    judge_errors: list[str] = []

    # Sample only claims that have a policy_verdict (otherwise judge has nothing to score)
    sample = [r for r in results if r.get("policy_verdict")][:max_sample]
    logger.info("LLM judge: scoring %d / %d claims", len(sample), len(results))

    for i, r in enumerate(sample, 1):
        cid = r.get("claim_id", "unknown")
        pv = r.get("policy_verdict") or {}
        gt = ground_truth.get(cid)

        logger.info("LLM judge [%d/%d] claim=%s", i, len(sample), cid)

        # ── Groundedness ──────────────────────────────────────────────────────
        try:
            policy_text = (
                f"Coverage: {pv.get('coverage_scope', '')}. "
                f"Exclusions: {pv.get('exclusions', '')}. "
                f"Policy Limit: {pv.get('policy_limit', '')}. "
                f"Remaining Limit: {pv.get('remaining_limit', '')}."
            )
            g_score = g_judge.score(policy_text=policy_text, verdict=pv)
            if g_score is not None:
                groundedness_scores.append(g_score.score)
                logger.debug("claim=%s | groundedness=%d", cid, g_score.score)
            else:
                judge_errors.append(f"Groundedness returned None for {cid}")
        except Exception as exc:
            judge_errors.append(f"Groundedness failed for {cid}: {exc}")
            logger.warning("Groundedness judge error for %s: %s", cid, exc)

        # Small pause between judge calls to avoid hitting RPM on free tier
        time.sleep(2)

        # ── Hallucination ─────────────────────────────────────────────────────
        try:
            reasoning = (r.get("final_decision") or {}).get("step_by_step_reasoning", "")
            if reasoning and gt:
                facts = (
                    f"Claim: {gt.incident_type}, Loss: ${gt.estimated_loss:,.2f}, "
                    f"Scenario: {gt.fraud_scenario}. "
                    f"Policy: remaining_limit={pv.get('remaining_limit')}, "
                    f"exclusions={str(pv.get('exclusions', ''))[:150]}. "
                    f"Fraud Risk: {r.get('fraud_report', {}).get('risk_score', 'Unknown')}. "
                    f"Errors: {r.get('errors', [])}."
                )
                h_report = h_judge.evaluate(facts=facts, reasoning=reasoning)
                if h_report is not None:
                    hallucination_rates.append(h_report.hallucination_rate)
                    logger.debug(
                        "claim=%s | hallucination_rate=%.3f", cid, h_report.hallucination_rate
                    )
                else:
                    judge_errors.append(f"Hallucination returned None for {cid}")
        except Exception as exc:
            judge_errors.append(f"Hallucination failed for {cid}: {exc}")
            logger.warning("Hallucination judge error for %s: %s", cid, exc)

        # Pause between claims
        time.sleep(2)

    return {
        "groundedness": {
            "mean_score": round(statistics.mean(groundedness_scores), 3) if groundedness_scores else None,
            "scores": groundedness_scores,
            "scale": "1 (unsupported) → 5 (fully grounded)",
            "claims_evaluated": len(groundedness_scores),
            "claims_skipped": len(sample) - len(groundedness_scores),
        },
        "hallucination": {
            "mean_rate": round(statistics.mean(hallucination_rates), 4) if hallucination_rates else None,
            "rates": hallucination_rates,
            "claims_evaluated": len(hallucination_rates),
            "claims_skipped": len(sample) - len(hallucination_rates),
        },
        "judge_errors": judge_errors,
        "sample_size": len(sample),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main orchestrator
# ══════════════════════════════════════════════════════════════════════════════

def run_evaluation(
    k: int = 1,
    run_chaos: bool = False,
    run_llm_judge: bool = True,
    output_dir: str = "evals/results",
    max_claims: int | None = None,
    explicit_claim_ids: list[str] | None = None,
    inter_claim_delay: float = 4.0,
    llm_judge_sample: int = 5,
) -> dict:
    """
    Full evaluation pipeline.

    Parameter defaults are intentionally conservative for speed:
      - k=1          (no consistency sweep — add --k 3 for full run)
      - run_chaos=False (skipped by default — add --chaos to enable)
      - llm_judge_sample=5  (score only 5 claims with the LLM judge)
    """
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("═" * 60)
    logger.info("  Car Insurance MAS — Evaluation Run  [%s]", run_id)
    logger.info(
        "  K=%d  chaos=%s  llm_judge=%s  delay=%.1fs  judge_sample=%d",
        k, run_chaos, run_llm_judge, inter_claim_delay, llm_judge_sample,
    )
    logger.info("═" * 60)

    # ── Load ground truth ──────────────────────────────────────────────────────
    logger.info("Loading ground truth from MongoDB…")
    try:
        ground_truth = load_ground_truth()
    except Exception as exc:
        logger.error("Ground truth load failed: %s", exc)
        sys.exit(1)

    if explicit_claim_ids:
        ground_truth = {
            cid: gt for cid, gt in ground_truth.items() if cid in explicit_claim_ids
        }
        logger.info("Filtered to %d explicit claim IDs.", len(ground_truth))
    elif max_claims:
        items = list(ground_truth.items())[:max_claims]
        ground_truth = dict(items)
        logger.info("Capped to first %d claims.", len(ground_truth))

    claim_ids = list(ground_truth.keys())
    logger.info("Evaluating %d claims.", len(claim_ids))

    # ── Single-pass runs ───────────────────────────────────────────────────────
    logger.info("Running single-pass evaluation…")
    results: list[dict] = []
    timing_records: list[dict] = []
    token_records: list[dict] = []

    for i, cid in enumerate(claim_ids, 1):
        logger.info("[%d/%d] Processing claim %s…", i, len(claim_ids), cid)
        result, elapsed = _timed_process(cid)
        results.append(result)
        timing_records.append({"claim_id": cid, "total_s": elapsed, "per_agent": {}})
        token_records.append({"claim_id": cid, "per_agent": _extract_token_usage(result)})

        if inter_claim_delay > 0 and i < len(claim_ids):
            actual = adaptive_sleep(inter_claim_delay)
            logger.debug("Slept %.1fs before next claim", actual)

    # ── Core metrics ───────────────────────────────────────────────────────────
    logger.info("Computing core metrics…")
    acc_metrics   = decision_accuracy(results, ground_truth)
    fraud_metrics = fraud_precision_recall_f1(results, ground_truth)
    stp_metrics   = stp_rate(results)
    completeness  = step_completeness(results)
    lat_metrics   = latency_stats(timing_records)
    tok_metrics   = token_efficiency(token_records)

    logger.info(
        "Core metrics → accuracy=%.2f%%  F1=%.2f%%  STP=%.2f%%",
        acc_metrics["accuracy"] * 100,
        fraud_metrics["f1"] * 100,
        stp_metrics["stp_rate"] * 100,
    )

    # ── Consistency (K runs) ───────────────────────────────────────────────────
    # Only run if K > 1 — consistency at K=1 is trivially 1.0 and wastes budget.
    multi_run: dict[str, list[dict]] = {}

    if k > 1:
        logger.info("Running consistency sweep (K=%d)…", k)
        # Seed multi_run with the first-pass results
        for r in results:
            cid = r.get("claim_id")
            if cid:
                multi_run[cid] = [r]

        for run_num in range(2, k + 1):
            logger.info("  Consistency run %d/%d…", run_num, k)
            for j, cid in enumerate(claim_ids, 1):
                result_k, _ = _timed_process(cid)
                multi_run[cid].append(result_k)
                if inter_claim_delay > 0 and j < len(claim_ids):
                    adaptive_sleep(inter_claim_delay)

        cons_metrics = consistency_score(multi_run)
    else:
        cons_metrics = {
            "mean_consistency": 1.0,
            "per_claim": {cid: 1.0 for cid in claim_ids},
            "k_runs": 1,
            "note": "K=1, consistency not measured. Use --k 3 for consistency sweep.",
        }
        logger.info("Skipping consistency (K=1).")

    # ── LLM-Judge ─────────────────────────────────────────────────────────────
    judge_metrics: dict = {}
    if run_llm_judge:
        logger.info(
            "Running LLM-as-Judge scoring (groundedness + hallucination, sample=%d)…",
            llm_judge_sample,
        )
        judge_metrics = _run_llm_judge(results, ground_truth, max_sample=llm_judge_sample)
    else:
        logger.info("LLM judge skipped (--no-llm-judge).")

    # ── Chaos testing ──────────────────────────────────────────────────────────
    chaos_report: dict = {}
    if run_chaos:
        logger.info("Running chaos/corruption sweep…")
        harness = ChaosHarness(process_fn=_process_claim)
        try:
            chaos_report = harness.run_corruption_sweep(
                ground_truth=ground_truth,
                corruption_levels=[0.0, 0.10, 0.20, 0.30],
                fields_to_corrupt=[
                    "PolicyNumber", "ClaimantName", "TotalEstimate", "LossDate"
                ],
            )
            chaos_summary = {
                k: v for k, v in chaos_report.items() if k != "per_level_details"
            }
        except Exception as exc:
            logger.error("Chaos testing failed: %s", exc)
            chaos_summary = {"error": str(exc)}
    else:
        chaos_summary = {}
        logger.info("Chaos testing skipped (pass --chaos to enable).")

    # ── Assemble report ────────────────────────────────────────────────────────
    report = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "claims_evaluated": len(claim_ids),
        "k_runs": k,
        "decision_accuracy": acc_metrics,
        "fraud_metrics": fraud_metrics,
        "stp": stp_metrics,
        "consistency": cons_metrics,
        "step_completeness": completeness,
        "latency": lat_metrics,
        "token_efficiency": tok_metrics,
        "llm_judge": judge_metrics,
        "chaos": chaos_summary,
        "scenario_breakdown": _scenario_breakdown(results, ground_truth),
    }

    # ── Save outputs ───────────────────────────────────────────────────────────
    json_path = out_dir / f"eval_report_{run_id}.json"
    chart_path = out_dir / f"eval_chart_{run_id}.png"

    save_json_report(report, json_path)
    try:
        build_report_chart(report, ground_truth, chart_path)
    except Exception as exc:
        logger.warning("Chart generation failed (non-fatal): %s", exc)

    _print_summary(report)
    logger.info("Results saved to %s", out_dir)
    return report


def _scenario_breakdown(
    results: list[dict], ground_truth: dict[str, GTRecord]
) -> dict:
    from collections import defaultdict
    scenario_stats: dict[str, dict] = defaultdict(lambda: {"correct": 0, "total": 0})

    for r in results:
        cid = r.get("claim_id")
        gt = ground_truth.get(cid)
        if not gt:
            continue
        fd = r.get("final_decision") or {}
        predicted = "Approved" if fd.get("approved") else "Denied"
        scenario = gt.fraud_scenario
        scenario_stats[scenario]["total"] += 1
        if predicted == gt.expected_decision:
            scenario_stats[scenario]["correct"] += 1

    return {
        scenario: {
            "accuracy": round(v["correct"] / v["total"], 4) if v["total"] else 0.0,
            "correct": v["correct"],
            "total": v["total"],
        }
        for scenario, v in scenario_stats.items()
    }


def _print_summary(report: dict) -> None:
    sep = "─" * 55
    print(f"\n{'═' * 55}")
    print(f"  EVALUATION COMPLETE  [{report['run_id']}]")
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
    cons = report["consistency"]
    print(f"  Consistency (K={cons.get('k_runs',1)})    : "
          f"{cons.get('mean_consistency',1.0)*100:.1f}%")
    sc = report["step_completeness"]
    print(f"  Step Completeness    : {sc.get('mean_completeness',0)*100:.1f}%")
    lat = report["latency"]
    if lat:
        print(f"  Mean Latency         : {lat.get('mean_latency_s',0):.2f}s  "
              f"(p95={lat.get('p95_latency_s',0):.2f}s)")
    jm = report.get("llm_judge", {})
    g = jm.get("groundedness", {})
    h = jm.get("hallucination", {})
    if g.get("mean_score") is not None:
        print(f"  Groundedness Score   : {g['mean_score']:.2f}/5.0  "
              f"(n={g.get('claims_evaluated', 0)})")
    if h.get("mean_rate") is not None:
        print(f"  Hallucination Rate   : {h['mean_rate']*100:.1f}%  "
              f"(n={h.get('claims_evaluated', 0)})")
    chaos = report.get("chaos", {})
    if chaos.get("degradation") is not None:
        print(f"  Chaos Degradation    : {chaos['degradation']*100:.1f}pp  "
              f"(0→30% corruption)")
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


# ── CLI entry point ────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Car Insurance MAS Evaluation Pipeline")
    p.add_argument(
        "--k", type=int, default=1,
        help="Consistency runs per claim (default 1 — use 3 for full sweep)",
    )
    p.add_argument(
        "--chaos", action="store_true",
        help="Enable chaos/corruption sweep (disabled by default for speed)",
    )
    p.add_argument(
        "--no-llm-judge", action="store_true",
        help="Skip LLM-as-Judge scoring",
    )
    p.add_argument(
        "--output-dir", default="evals/results",
        help="Output directory for reports",
    )
    p.add_argument(
        "--claims", type=int, default=None,
        help="Process first N claims only",
    )
    p.add_argument(
        "--claim-ids", nargs="+", default=None,
        help="Explicit claim IDs to process",
    )
    p.add_argument(
        "--delay", type=float, default=4.0,
        help=(
            "Seconds to sleep between claims (default 4.0). "
            "Each claim fires 2 concurrent LLM calls (parallel_analysis), "
            "doubling effective RPM. Increase if you hit 429s; set 0 to disable."
        ),
    )
    p.add_argument(
        "--llm-judge-sample", type=int, default=5,
        help="Max claims to score with LLM judge (default 5)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_evaluation(
        k=args.k,
        run_chaos=args.chaos,
        run_llm_judge=not args.no_llm_judge,
        output_dir=args.output_dir,
        max_claims=args.claims,
        explicit_claim_ids=args.claim_ids,
        inter_claim_delay=args.delay,
        llm_judge_sample=args.llm_judge_sample,
    )