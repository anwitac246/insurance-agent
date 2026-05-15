"""
run_evals.py
------------
Top-level evaluation script for the Car Insurance MAS.

Run
---
    python -m evals.run_evals [--k 3] [--no-chaos] [--no-llm-judge]
                              [--output-dir evals/results] [--claims N]
                              [--claim-ids id1 id2 ...] [--delay 0]
                              [--concurrency 5]

Speed improvements in this version
------------------------------------
- Default --delay is now 0 (was 4). The original delay was copied from Groq
  free-tier rate-limit guidance. Ollama runs locally with no rate limits;
  the delay was pure dead time (~3.5 min wasted on 50 claims).

- --concurrency N (default 5): claims are now processed in concurrent batches
  using asyncio.gather(). Each claim fires 3 LLM calls (2 parallel via
  parallel_analysis + 1 for decision). With Ollama, concurrent requests are
  queued server-side and processed as fast as the GPU allows — throughput is
  significantly higher than strict sequential processing.

  Expected speedup:
    Sequential (old):  50 claims × 160s = ~2.2 hours
    Concurrent ×5:     10 batches × 160s / batch ≈ ~27 min
    Concurrent ×10:    5  batches × 160s / batch ≈ ~13 min

  Tune --concurrency to your GPU VRAM. llama3:latest at 4-bit quantization
  fits in ~8GB VRAM; 5 concurrent requests is safe for 24GB cards.
  Use --concurrency 3 if you see OOM errors or severe slowdowns.

- LLM judge sample unchanged at 5 (local Ollama is slower for judge calls
  since they compete with pipeline calls; keep low during main eval).

Arguments
---------
--k              Consistency runs per claim (default 1).
--chaos          Enable OCR corruption sweep.
--no-llm-judge   Skip groundedness + hallucination scoring.
--output-dir     Output directory (default: evals/results).
--claims N       Evaluate first N claims only.
--claim-ids      Explicit claim IDs.
--delay          Seconds between BATCHES (default 0). Only useful if hitting
                 an external API with rate limits. Set >0 for Groq/OpenAI.
--concurrency    Claims processed in parallel per batch (default 5).
--llm-judge-sample  Max claims scored by LLM judge (default 5).
"""

from __future__ import annotations

import argparse
import asyncio
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
from src.tools.llm_client import adaptive_sleep

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
    logger.error("Cannot import src.main.process_claim: %s", exc)
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


async def _async_timed_process(claim_id: str) -> tuple[dict, float]:
    """
    Async wrapper that runs _timed_process in a thread pool so that
    asyncio.gather() can overlap multiple claims concurrently.

    process_claim() internally bridges sync→async correctly via its own
    ThreadPoolExecutor in graph.py, so running it from a thread here is safe.
    """
    loop = asyncio.get_event_loop()
    t0 = time.perf_counter()
    try:
        result = await loop.run_in_executor(None, _process_claim, claim_id)
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


async def _process_batch(claim_ids: list[str], batch_num: int, total_batches: int) -> list[tuple[dict, float]]:
    """Process a batch of claims concurrently."""
    logger.info(
        "Batch [%d/%d]: processing %d claims concurrently…",
        batch_num, total_batches, len(claim_ids),
    )
    tasks = [_async_timed_process(cid) for cid in claim_ids]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Handle any tasks that raised exceptions (already handled inside _async_timed_process,
    # but gather with return_exceptions=True catches anything that slips through)
    processed = []
    for cid, res in zip(claim_ids, results):
        if isinstance(res, Exception):
            logger.error("Claim %s batch task raised: %s", cid, res)
            processed.append(({
                "claim_id": cid,
                "final_decision": {"approved": False, "denial_reason": str(res)},
                "errors": [str(res)],
                "final_payout": 0.0,
            }, 0.0))
        else:
            processed.append(res)
    return processed


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


# ── LLM-Judge scoring ──────────────────────────────────────────────────────────

def _build_facts_string(r: dict, gt: GTRecord) -> str:
    pv = r.get("policy_verdict") or {}
    fr = r.get("fraud_report") or {}

    estimated_loss   = gt.estimated_loss
    remaining_limit  = float(pv.get("remaining_limit", 0))
    deductible       = float(pv.get("deductible", 0))
    aggregate_limit  = float(pv.get("aggregate_limit", 0))
    total_paid       = float(pv.get("total_historical_payout", 0))
    expected_payout  = max(0.0, min(estimated_loss - deductible, remaining_limit))

    anomalies        = fr.get("anomalies", [])
    primary_anomaly  = anomalies[0] if anomalies else "none"
    fraud_flags      = (
        f"frequent_claims={fr.get('frequent_claims_flag')}, "
        f"collusion={fr.get('collusion_flag')}, "
        f"staging={fr.get('staging_flag')}"
    )

    exclusion_reason = pv.get("exclusion_reason", "none")
    coverage_scope   = (pv.get("coverage_scope") or "N/A")[:200]
    denial_signals   = "; ".join(r.get("errors", [])) or "none"

    return (
        f"Claim ID: {gt.claim_id}. "
        f"Incident type: {gt.incident_type}. "
        f"Estimated loss: ${estimated_loss:,.2f}. "
        f"Deductible: ${deductible:,.2f}. "
        f"Remaining policy limit: ${remaining_limit:,.2f} "
        f"(aggregate=${aggregate_limit:,.2f}, total paid=${total_paid:,.2f}). "
        f"Expected payout if approved: ${expected_payout:,.2f} "
        f"(formula: min(loss - deductible, remaining_limit)). "
        f"Policy coverage: incident_covered={pv.get('incident_covered')}, "
        f"exclusion_triggered={pv.get('exclusion_triggered')}, "
        f"exclusion_reason={exclusion_reason!r}. "
        f"Coverage scope excerpt: {coverage_scope}. "
        f"Fraud risk: {fr.get('risk_score', 'Unknown')}. "
        f"Fraud flags: {fraud_flags}. "
        f"Primary anomaly: {primary_anomaly}. "
        f"Active denial signals: {denial_signals}."
    )


def _run_llm_judge(
    results: list[dict],
    ground_truth: dict[str, GTRecord],
    max_sample: int = 5,
) -> dict:
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

    sample = [r for r in results if r.get("policy_verdict")][:max_sample]
    logger.info("LLM judge: scoring %d / %d claims", len(sample), len(results))

    for i, r in enumerate(sample, 1):
        cid = r.get("claim_id", "unknown")
        pv = r.get("policy_verdict") or {}
        gt = ground_truth.get(cid)

        logger.info("LLM judge [%d/%d] claim=%s", i, len(sample), cid)

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
            else:
                judge_errors.append(f"Groundedness returned None for {cid}")
        except Exception as exc:
            judge_errors.append(f"Groundedness failed for {cid}: {exc}")
            logger.warning("Groundedness judge error for %s: %s", cid, exc)

        time.sleep(1)

        try:
            reasoning = (r.get("final_decision") or {}).get("step_by_step_reasoning", "")
            if reasoning and gt:
                facts = _build_facts_string(r, gt)
                h_report = h_judge.evaluate(facts=facts, reasoning=reasoning)
                if h_report is not None:
                    hallucination_rates.append(h_report.hallucination_rate)
                else:
                    judge_errors.append(f"Hallucination returned None for {cid}")
        except Exception as exc:
            judge_errors.append(f"Hallucination failed for {cid}: {exc}")
            logger.warning("Hallucination judge error for %s: %s", cid, exc)

        time.sleep(1)

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
    inter_claim_delay: float = 0.0,   # FIX: default 0 (was 4) — Ollama is local
    llm_judge_sample: int = 5,
    concurrency: int = 5,             # NEW: concurrent claims per batch
) -> dict:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("═" * 60)
    logger.info("  Car Insurance MAS — Evaluation Run  [%s]", run_id)
    logger.info(
        "  K=%d  chaos=%s  llm_judge=%s  concurrency=%d  delay=%.1fs  judge_sample=%d",
        k, run_chaos, run_llm_judge, concurrency, inter_claim_delay, llm_judge_sample,
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
        ground_truth = {cid: gt for cid, gt in ground_truth.items() if cid in explicit_claim_ids}
        logger.info("Filtered to %d explicit claim IDs.", len(ground_truth))
    elif max_claims:
        items = list(ground_truth.items())[:max_claims]
        ground_truth = dict(items)
        logger.info("Capped to first %d claims.", len(ground_truth))

    claim_ids = list(ground_truth.keys())
    logger.info("Evaluating %d claims (concurrency=%d).", len(claim_ids), concurrency)

    # ── Concurrent batch processing ────────────────────────────────────────────
    logger.info("Running single-pass evaluation (concurrent batches)…")
    results: list[dict] = []
    timing_records: list[dict] = []
    token_records: list[dict] = []

    # Split claim_ids into batches of size `concurrency`
    batches = [
        claim_ids[i:i + concurrency]
        for i in range(0, len(claim_ids), concurrency)
    ]
    total_batches = len(batches)

    wall_t0 = time.perf_counter()

    for batch_idx, batch in enumerate(batches, 1):
        batch_results = asyncio.run(_process_batch(batch, batch_idx, total_batches))

        for cid, (result, elapsed) in zip(batch, batch_results):
            results.append(result)
            timing_records.append({"claim_id": cid, "total_s": elapsed, "per_agent": {}})
            token_records.append({"claim_id": cid, "per_agent": _extract_token_usage(result)})

            fd = result.get("final_decision") or {}
            logger.info(
                "  claim=%s | approved=%s | payout=$%.2f | %.1fs",
                cid,
                fd.get("approved"),
                result.get("final_payout", 0.0),
                elapsed,
            )

        # Inter-BATCH delay (only relevant for external API rate limits)
        if inter_claim_delay > 0 and batch_idx < total_batches:
            logger.info("Sleeping %.1fs between batches…", inter_claim_delay)
            time.sleep(inter_claim_delay)

    total_wall = time.perf_counter() - wall_t0
    logger.info(
        "All %d claims processed in %.1fs (%.1fs/claim average, concurrency=%d)",
        len(claim_ids), total_wall, total_wall / len(claim_ids), concurrency,
    )

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
    multi_run: dict[str, list[dict]] = {}

    if k > 1:
        logger.info("Running consistency sweep (K=%d)…", k)
        for r in results:
            cid = r.get("claim_id")
            if cid:
                multi_run[cid] = [r]

        for run_num in range(2, k + 1):
            logger.info("  Consistency run %d/%d…", run_num, k)
            for batch_idx, batch in enumerate(batches, 1):
                batch_results_k = asyncio.run(
                    _process_batch(batch, batch_idx, total_batches)
                )
                for cid, (result_k, _) in zip(batch, batch_results_k):
                    multi_run[cid].append(result_k)

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
            "Running LLM-as-Judge scoring (sample=%d)…", llm_judge_sample
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
        "concurrency": concurrency,
        "total_wall_seconds": round(total_wall, 1),
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
    wall = report.get("total_wall_seconds", 0)
    if wall:
        per_claim = wall / max(report["claims_evaluated"], 1)
        print(f"  Wall time            : {wall:.0f}s  ({per_claim:.1f}s/claim, concurrency={report.get('concurrency',1)})")
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


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Car Insurance MAS Evaluation Pipeline")
    p.add_argument("--k", type=int, default=1,
                   help="Consistency runs per claim (default 1)")
    p.add_argument("--chaos", action="store_true",
                   help="Enable chaos/corruption sweep")
    p.add_argument("--no-llm-judge", action="store_true",
                   help="Skip LLM-as-Judge scoring")
    p.add_argument("--output-dir", default="evals/results",
                   help="Output directory for reports")
    p.add_argument("--claims", type=int, default=None,
                   help="Process first N claims only")
    p.add_argument("--claim-ids", nargs="+", default=None,
                   help="Explicit claim IDs to process")
    p.add_argument(
        "--delay", type=float, default=0.0,
        help=(
            "Seconds to sleep between BATCHES (default 0). "
            "Only needed for external API rate limits (Groq, OpenAI). "
            "Ollama is local — leave at 0."
        ),
    )
    p.add_argument(
        "--concurrency", type=int, default=5,
        help=(
            "Number of claims processed in parallel per batch (default 5). "
            "Tune to your GPU VRAM. Reduce to 3 if you see OOM errors."
        ),
    )
    p.add_argument("--llm-judge-sample", type=int, default=5,
                   help="Max claims scored by LLM judge (default 5)")
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
        concurrency=args.concurrency,
    )