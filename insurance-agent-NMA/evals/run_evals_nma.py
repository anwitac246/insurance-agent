import sys
import os
import asyncio
import time
import logging
from datetime import datetime
from tqdm import tqdm

# Setup sys.path to access both root and NMA directories
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
nma_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
if nma_dir not in sys.path:
    sys.path.insert(0, nma_dir)

from nma_src.tools.context_fetcher import fetch_nma_context
from nma_src.agents.nma_agent import arun_nma_agent
from src.tools.mongo_client import get_db

from evals.reporter import save_json_report, build_report_chart
from evals.ground_truth import load_ground_truth
from evals.metrics import (
    decision_accuracy,
    fraud_precision_recall_f1,
    stp_rate,
    latency_stats
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

async def run_evals_nma():
    db = get_db()
    claims_cursor = db.Active_Claims.find()
    claims = await asyncio.to_thread(list, claims_cursor)
    
    ground_truth = load_ground_truth()
    
    results = []
    latencies = []
    start_run = time.perf_counter()
    
    # Process sequentially to avoid Groq rate limits, similar to MAS
    for i, c in enumerate(claims, 1):
        cid = c["claim_id"]
        logger.info("Processing [%d/%d] %s", i, len(claims), cid)
        
        t0 = time.perf_counter()
        try:
            ctx = await fetch_nma_context(cid)
            res = await arun_nma_agent(ctx)
            
            lat = time.perf_counter() - t0
            latencies.append(lat)
            
            # Format result to match MAS output structure so metrics calculations work
            results.append({
                "claim_id": cid,
                "latency_seconds": lat,
                "final_decision": {
                    "approved": res.approved,
                    "final_payout": res.final_payout,
                    "step_by_step_reasoning": res.step_by_step_reasoning
                },
                "fraud_report": {
                    "risk_score": "High" if res.fraud_risk_score >= 8 else "Medium" if res.fraud_risk_score >= 4 else "Low",
                    "anomalies": [res.fraud_anomalies] if res.fraud_anomalies != "None" else []
                },
                "policy_verdict": {
                    "incident_covered": res.incident_covered,
                    "exclusion_triggered": res.exclusion_triggered,
                    "exclusion_reason": res.exclusion_reason
                }
            })
            
            logger.info("Claim %s complete in %.2fs. Approved=%s", cid, lat, res.approved)
            
            # Rate limit backoff
            await asyncio.sleep(3.0)
            
        except Exception as e:
            logger.error("Failed claim %s: %s", cid, e)
            
    total_time = time.perf_counter() - start_run
    
    # Calculate metrics using the same functions as MAS
    acc_metrics = decision_accuracy(results, ground_truth)
    fraud_metrics = fraud_precision_recall_f1(results, ground_truth)
    stp_metrics = stp_rate(results)
    
    timing_records = [{"claim_id": r["claim_id"], "total_s": r["latency_seconds"]} for r in results]
    lat_metrics = latency_stats(timing_records)
    tok_metrics = {} # groq_client tracks globally, NMA script doesn't parse it per claim yet
    
    report = {
        "run_id": f"NMA_{int(time.time())}",
        "timestamp": datetime.now().isoformat(),
        "claims_evaluated": len(results),
        "decision_accuracy": acc_metrics,
        "fraud_metrics": fraud_metrics,
        "stp": stp_metrics,
        "latency": lat_metrics,
        "token_efficiency": tok_metrics,
        "total_time_seconds": round(total_time, 2)
    }
    
    out_dir = os.path.join(root_dir, "evals", "results")
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, f"eval_report_{report['run_id']}.json")
    save_json_report(report, json_path)
    
    print("\n=== NMA EVALUATION COMPLETE ===")
    print(f"Accuracy: {acc_metrics.get('accuracy', 0)*100:.1f}%")
    print(f"Average Latency: {lat_metrics.get('mean_latency_s', 0):.2f}s")
    print(f"Total Tokens: {tok_metrics.get('total_tokens', 0)}")
    print(f"Report saved to {json_path}")

if __name__ == "__main__":
    asyncio.run(run_evals_nma())
