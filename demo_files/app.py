"""
app.py
------
Flask backend for the Insurance MAS Demo.

Folder structure (from repo root insurance-agent/):
    demo_files/
        app.py              ← this file
        templates/
            index.html
        DEMO.md
        run_demo.sh
    evals/
    insurance-agent-NMA/
        nma_src/
            agents/nma_agent.py
            tools/context_fetcher.py
            schemas/nma_schema.py
    src/
        main.py
        agents/
        graph/
        tools/

Run from the repo root:
    python demo_files/app.py
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path

# ── Resolve paths ─────────────────────────────────────────────────────────────
# app.py is at: <repo_root>/demo_files/app.py
DEMO_DIR = Path(__file__).resolve().parent          # .../demo_files/
ROOT     = DEMO_DIR.parent                          # .../insurance-agent/
NMA_DIR  = ROOT / "insurance-agent-NMA"             # .../insurance-agent-NMA/

# Add repo root so `src.*` and `evals.*` imports resolve
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Add NMA dir so `nma_src.*` imports resolve (hyphen in dir name prevents
# normal Python package import, so we add it to sys.path directly)
if str(NMA_DIR) not in sys.path:
    sys.path.insert(0, str(NMA_DIR))

# Load .env from repo root
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("demo.app")

# template_folder uses absolute path so it works regardless of cwd
app = Flask(__name__, template_folder=str(DEMO_DIR / "templates"))
CORS(app)


# ── Lazy pipeline imports ─────────────────────────────────────────────────────
# Deferred so Flask starts immediately; pipelines import on first request.

def _get_mas():
    from src.main import process_claim
    return process_claim


def _get_nma_context():
    # NMA dir is already on sys.path, so nma_src.* resolves directly
    from nma_src.tools.context_fetcher import fetch_nma_context
    return fetch_nma_context


def _get_nma_agent():
    from nma_src.agents.nma_agent import arun_nma_agent
    return arun_nma_agent


# ── DB helpers ────────────────────────────────────────────────────────────────

def _get_db():
    from src.tools.mongo_client import get_db
    return get_db()


def _inject_claim(narrative, incident_type, estimated_loss, repair_shop,
                  ocr_estimate=None):
    """
    Insert a temporary demo claim into Active_Claims, borrowing the first
    existing customer so the Pinecone policy lookup works end-to-end.
    Returns the new claim_id string.
    """
    db = _get_db()

    existing = db["Customer_Profiles"].find_one({})
    if not existing:
        raise RuntimeError(
            "No customers in DB — run: python scripts/seed_data.py"
        )

    customer_id = existing["customer_id"]
    policy_id   = existing["policy_id"]
    claimant    = existing.get("full_name", "Demo User")
    claim_id    = str(uuid.uuid4())
    loss_date   = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d")

    db["Active_Claims"].insert_one({
        "claim_id":       claim_id,
        "policy_id":      policy_id,
        "customer_id":    customer_id,
        "incident_type":  incident_type,
        "incident_date":  loss_date,
        "narrative":      narrative,
        "ocr_extraction": {
            "PolicyNumber":   policy_id[:8].upper(),
            "ClaimantName":   claimant,
            "LossDate":       loss_date,
            "RepairShopName": repair_shop,
            "TotalEstimate":  ocr_estimate if ocr_estimate is not None else estimated_loss,
        },
        "estimated_loss": estimated_loss,
        "fraud_scenario": "demo",
    })
    logger.info("Injected demo claim %s", claim_id)
    return claim_id


def _cleanup_claim(claim_id: str):
    try:
        _get_db()["Active_Claims"].delete_one({"claim_id": claim_id})
        logger.info("Cleaned up claim %s", claim_id)
    except Exception as exc:
        logger.warning("Cleanup failed for %s: %s", claim_id, exc)


# ── JSON serialization ────────────────────────────────────────────────────────

def _serialize(obj):
    """
    Recursively make a value JSON-safe.
    - Strips MongoDB _id fields (ObjectId is not serializable)
    - Converts Enum, Pydantic models, and anything else to str
    """
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items() if k != "_id"}
    if isinstance(obj, list):
        return [_serialize(i) for i in obj]
    if isinstance(obj, bool):   # bool must come before int (bool subclasses int)
        return obj
    if isinstance(obj, (int, float, str)) or obj is None:
        return obj
    return str(obj)


# ── NMA result shaping ────────────────────────────────────────────────────────

def _shape_nma_result(claim_id, res, ctx):
    """Convert NMAOutput + context dict into the same shape as a MAS result."""
    pol  = ctx.get("policy", {})
    agg  = float(pol.get("aggregate_limit", 0))
    paid = float(pol.get("total_historical_payout", 0))

    return {
        "claim_id":     claim_id,
        "final_payout": res.final_payout,
        "final_decision": {
            "approved":               res.approved,
            "step_by_step_reasoning": res.step_by_step_reasoning,
            "denial_reason":          "" if res.approved else res.step_by_step_reasoning,
        },
        "policy_verdict": {
            "incident_covered":        res.incident_covered,
            "exclusion_triggered":     res.exclusion_triggered,
            "exclusion_reason":        res.exclusion_reason,
            "coverage_scope":          pol.get("coverage_scope", ""),
            "exclusions":              pol.get("exclusions", ""),
            "policy_limit":            float(pol.get("policy_limit", 0)),
            "aggregate_limit":         agg,
            "deductible":              float(pol.get("deductible", 0)),
            "total_historical_payout": paid,
            "remaining_limit":         agg - paid,
        },
        "fraud_report": {
            "risk_score": (
                "High"   if res.fraud_risk_score >= 8 else
                "Medium" if res.fraud_risk_score >= 4 else
                "Low"
            ),
            "frequent_claims_flag": False,
            "collusion_flag":       False,
            "staging_flag":         False,
            "anomalies": (
                [res.fraud_anomalies]
                if res.fraud_anomalies and res.fraud_anomalies != "None"
                else []
            ),
            "reasoning": res.step_by_step_reasoning,
        },
        "errors":   [],
        "warnings": [],
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "timestamp": datetime.now().isoformat()})


@app.route("/api/run_mas", methods=["POST"])
def run_mas():
    data     = request.get_json()
    t0       = time.perf_counter()
    claim_id = None
    try:
        claim_id = _inject_claim(
            narrative      = data["narrative"],
            incident_type  = data["incident_type"],
            estimated_loss = float(data["estimated_loss"]),
            repair_shop    = data["repair_shop"],
            ocr_estimate   = float(data["ocr_estimate"]) if data.get("ocr_estimate") else None,
        )
        result  = _get_mas()(claim_id)
        elapsed = round(time.perf_counter() - t0, 2)
        return jsonify({"ok": True, "elapsed_s": elapsed, "result": _serialize(result)})
    except Exception as exc:
        logger.exception("MAS pipeline error")
        return jsonify({"ok": False, "error": str(exc)}), 500
    finally:
        if claim_id:
            _cleanup_claim(claim_id)


@app.route("/api/run_nma", methods=["POST"])
def run_nma():
    data     = request.get_json()
    t0       = time.perf_counter()
    claim_id = None
    try:
        claim_id = _inject_claim(
            narrative      = data["narrative"],
            incident_type  = data["incident_type"],
            estimated_loss = float(data["estimated_loss"]),
            repair_shop    = data["repair_shop"],
            ocr_estimate   = float(data["ocr_estimate"]) if data.get("ocr_estimate") else None,
        )

        async def _run():
            ctx = await _get_nma_context()(claim_id)
            res = await _get_nma_agent()(ctx)
            return ctx, res

        ctx, res = asyncio.run(_run())
        elapsed  = round(time.perf_counter() - t0, 2)
        result   = _shape_nma_result(claim_id, res, ctx)
        return jsonify({"ok": True, "elapsed_s": elapsed, "result": _serialize(result)})
    except Exception as exc:
        logger.exception("NMA pipeline error")
        return jsonify({"ok": False, "error": str(exc)}), 500
    finally:
        if claim_id:
            _cleanup_claim(claim_id)


@app.route("/api/run_both", methods=["POST"])
def run_both():
    data     = request.get_json()
    t0       = time.perf_counter()
    claim_id = None
    try:
        claim_id = _inject_claim(
            narrative      = data["narrative"],
            incident_type  = data["incident_type"],
            estimated_loss = float(data["estimated_loss"]),
            repair_shop    = data["repair_shop"],
            ocr_estimate   = float(data["ocr_estimate"]) if data.get("ocr_estimate") else None,
        )

        # ── MAS (synchronous LangGraph pipeline) ──────────────────────────────
        mas_t0      = time.perf_counter()
        mas_result  = _get_mas()(claim_id)
        mas_elapsed = round(time.perf_counter() - mas_t0, 2)

        # ── NMA (async) ───────────────────────────────────────────────────────
        nma_t0 = time.perf_counter()

        async def _run_nma():
            ctx = await _get_nma_context()(claim_id)
            res = await _get_nma_agent()(ctx)
            return ctx, res

        ctx, res    = asyncio.run(_run_nma())
        nma_elapsed = round(time.perf_counter() - nma_t0, 2)
        nma_result  = _shape_nma_result(claim_id, res, ctx)

        return jsonify({
            "ok":              True,
            "total_elapsed_s": round(time.perf_counter() - t0, 2),
            "mas": {"elapsed_s": mas_elapsed, "result": _serialize(mas_result)},
            "nma": {"elapsed_s": nma_elapsed, "result": _serialize(nma_result)},
        })
    except Exception as exc:
        logger.exception("run_both error")
        return jsonify({"ok": False, "error": str(exc)}), 500
    finally:
        if claim_id:
            _cleanup_claim(claim_id)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    logger.info("=" * 55)
    logger.info("  Insurance MAS Demo  →  http://localhost:%d", port)
    logger.info("  ROOT    : %s", ROOT)
    logger.info("  NMA_DIR : %s", NMA_DIR)
    logger.info("=" * 55)
    app.run(host="0.0.0.0", port=port, debug=False)