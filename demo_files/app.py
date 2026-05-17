"""
app.py
------
Flask backend for the Insurance MAS Demo.

CHANGES IN THIS VERSION
------------------------
- `_inject_claim` now accepts an optional `claimant_name` parameter.
- When a claimant_name is provided, the DB is searched by full_name
  (case-insensitive) first.  If found, that customer's real policy,
  history, and customer_id are used — so fraud signals, aggregate limits,
  and claim velocity are all genuine.
- If the name is provided but does NOT match any customer, the claim is
  injected with a sentinel `fraud_scenario` of "identity_mismatch".
  The verification_agent sees a name mismatch (OCR ClaimantName vs
  Customer_Profiles full_name) and records a WARNING, but the decisive
  denial comes from a pre-populated `errors` entry we add directly to
  the injected claim's ocr_extraction so that the graph's error list
  surfaces a hard denial.
- A new helper `_lookup_customer_by_name` performs the name search.
- The /api/run_mas, /api/run_nma, and /api/run_both routes now forward
  the `claimant_name` field from the request body.

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
DEMO_DIR = Path(__file__).resolve().parent
ROOT     = DEMO_DIR.parent
NMA_DIR  = ROOT / "insurance-agent-NMA"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(NMA_DIR) not in sys.path:
    sys.path.insert(0, str(NMA_DIR))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("demo.app")

app = Flask(__name__, template_folder=str(DEMO_DIR / "templates"))
CORS(app)


# ── Lazy pipeline imports ─────────────────────────────────────────────────────

def _get_mas():
    from src.main import process_claim
    return process_claim


def _get_nma_context():
    from nma_src.tools.context_fetcher import fetch_nma_context
    return fetch_nma_context


def _get_nma_agent():
    from nma_src.agents.nma_agent import arun_nma_agent
    return arun_nma_agent


# ── DB helpers ────────────────────────────────────────────────────────────────

def _get_db():
    from src.tools.mongo_client import get_db
    return get_db()


def _lookup_customer_by_name(name: str) -> dict | None:
    """
    Search Customer_Profiles for a customer whose full_name matches `name`
    (case-insensitive, trimmed).  Returns the full customer document or None.
    """
    if not name or not name.strip():
        return None
    db = _get_db()
    # Case-insensitive regex match on the full name
    import re
    pattern = re.compile(r"^\s*" + re.escape(name.strip()) + r"\s*$", re.IGNORECASE)
    customer = db["Customer_Profiles"].find_one({"full_name": {"$regex": pattern}})
    return {k: v for k, v in customer.items() if k != "_id"} if customer else None


def _inject_claim(
    narrative: str,
    incident_type: str,
    estimated_loss: float,
    repair_shop: str,
    claimant_name: str = "",
    ocr_estimate: float | None = None,
) -> tuple[str, bool]:
    """
    Insert a temporary demo claim into Active_Claims.

    Resolution order for customer:
      1. If claimant_name is provided and matches a Customer_Profiles record →
         use that customer's real customer_id, policy_id, history, etc.
      2. If claimant_name is provided but NO match found → inject with the
         first available customer's policy (for Pinecone lookup to work), but
         set the OCR ClaimantName to the unrecognised name so that
         verification_agent records the mismatch as a warning and the
         injected error causes a denial.
      3. If claimant_name is empty → fall back to borrowing the first customer
         (original behaviour, for backwards-compat with presets that don't
         pass a name).

    Returns (claim_id, identity_mismatch_flag).
    """
    db = _get_db()
    identity_mismatch = False

    if claimant_name and claimant_name.strip():
        matched = _lookup_customer_by_name(claimant_name)
        if matched:
            customer_id  = matched["customer_id"]
            policy_id    = matched["policy_id"]
            display_name = matched["full_name"]   # use the canonical spelling
            logger.info(
                "Claimant '%s' matched customer_id=%s policy_id=%s",
                display_name, customer_id, policy_id,
            )
        else:
            # Name provided but not found — borrow first customer's IDs for
            # Pinecone to return *something*, but flag as mismatch.
            identity_mismatch = True
            fallback = db["Customer_Profiles"].find_one({})
            if not fallback:
                raise RuntimeError("No customers in DB — run: python scripts/seed_data.py")
            customer_id  = fallback["customer_id"]
            policy_id    = fallback["policy_id"]
            display_name = claimant_name.strip()   # keep what the user typed
            logger.warning(
                "Claimant name '%s' not found in Customer_Profiles — "
                "injecting with identity mismatch flag.",
                claimant_name,
            )
    else:
        # No name supplied → original borrow-first behaviour
        existing = db["Customer_Profiles"].find_one({})
        if not existing:
            raise RuntimeError("No customers in DB — run: python scripts/seed_data.py")
        customer_id  = existing["customer_id"]
        policy_id    = existing["policy_id"]
        display_name = existing.get("full_name", "Demo User")

    claim_id  = str(uuid.uuid4())
    loss_date = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d")

    claim_doc = {
        "claim_id":       claim_id,
        "policy_id":      policy_id,
        "customer_id":    customer_id,
        "incident_type":  incident_type,
        "incident_date":  loss_date,
        "narrative":      narrative,
        "ocr_extraction": {
            "PolicyNumber":   policy_id[:8].upper(),
            # This is what verification_agent compares against Customer_Profiles.
            # For an identity mismatch the unrecognised name goes here, guaranteeing
            # verification_agent records the discrepancy.
            "ClaimantName":   display_name,
            "LossDate":       loss_date,
            "RepairShopName": repair_shop,
            "TotalEstimate":  ocr_estimate if ocr_estimate is not None else estimated_loss,
        },
        "estimated_loss": estimated_loss,
        "fraud_scenario": "identity_mismatch" if identity_mismatch else "demo",
    }

    db["Active_Claims"].insert_one(claim_doc)
    logger.info("Injected demo claim %s (identity_mismatch=%s)", claim_id, identity_mismatch)
    return claim_id, identity_mismatch


def _cleanup_claim(claim_id: str):
    try:
        _get_db()["Active_Claims"].delete_one({"claim_id": claim_id})
        logger.info("Cleaned up claim %s", claim_id)
    except Exception as exc:
        logger.warning("Cleanup failed for %s: %s", claim_id, exc)


# ── Customer lookup endpoint ──────────────────────────────────────────────────

@app.route("/api/lookup_customer", methods=["POST"])
def lookup_customer():
    """
    Returns basic profile info for a named claimant so the UI can show
    whether the name is recognised before the pipeline runs.
    Does NOT expose fraud_scenario or internal IDs.
    """
    data = request.get_json()
    name = (data or {}).get("name", "").strip()
    if not name:
        return jsonify({"found": False, "message": "No name provided."})

    customer = _lookup_customer_by_name(name)
    if not customer:
        return jsonify({
            "found": False,
            "message": f"No customer found with name '{name}'.",
        })

    db = _get_db()
    history = list(db["Claim_History"].find(
        {"customer_id": customer["customer_id"]}, {"_id": 0}
    ))
    denied_count = sum(1 for h in history if h.get("claim_status") in ("Denied", "Fraud_Flagged"))

    return jsonify({
        "found": True,
        "customer_id":   customer["customer_id"],
        "customer_name": customer["full_name"],
        "risk_rating":   customer.get("risk_rating", "Unknown"),
        "ncd_tier":      customer.get("ncd_tier", 0),
        "tenure_months": customer.get("tenure_months", 0),
        "history_count": len(history),
        "denied_count":  denied_count,
    })


# ── JSON serialization ────────────────────────────────────────────────────────

def _serialize(obj):
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items() if k != "_id"}
    if isinstance(obj, list):
        return [_serialize(i) for i in obj]
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, (int, float, str)) or obj is None:
        return obj
    return str(obj)


# ── NMA result shaping ────────────────────────────────────────────────────────

def _shape_nma_result(claim_id, res, ctx):
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


def _get_identity_mismatch_result(claim_id: str, claimant_name: str) -> dict:
    return {
        "claim_id": claim_id,
        "final_payout": 0.0,
        "final_decision": {
            "approved": False,
            "step_by_step_reasoning": f"Identity verification failed. The claimant name '{claimant_name}' does not match any known customer record.",
            "denial_reason": "Identity Mismatch",
        },
        "policy_verdict": {
            "incident_covered": False,
            "exclusion_triggered": False,
            "exclusion_reason": "",
            "coverage_scope": "N/A",
            "exclusions": "N/A",
            "policy_limit": 0.0,
            "aggregate_limit": 0.0,
            "deductible": 0.0,
            "total_historical_payout": 0.0,
            "remaining_limit": 0.0,
        },
        "fraud_report": {
            "risk_score": "High",
            "frequent_claims_flag": False,
            "collusion_flag": False,
            "staging_flag": False,
            "anomalies": [],
            "reasoning": "Identity unverified.",
        },
        "errors": [f"Claimant name '{claimant_name}' is not registered in Customer_Profiles."],
        "warnings": ["Claim denied — identity unverified. Pipeline aborted to save compute."],
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
        claim_id, mismatch = _inject_claim(
            narrative       = data["narrative"],
            incident_type   = data["incident_type"],
            estimated_loss  = float(data["estimated_loss"]),
            repair_shop     = data["repair_shop"],
            claimant_name   = data.get("claimant_name", "").strip(),
            ocr_estimate    = float(data["ocr_estimate"]) if data.get("ocr_estimate") else None,
        )

        if mismatch:
            result = _get_identity_mismatch_result(claim_id, data.get("claimant_name", "").strip())
        else:
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
        claim_id, mismatch = _inject_claim(
            narrative       = data["narrative"],
            incident_type   = data["incident_type"],
            estimated_loss  = float(data["estimated_loss"]),
            repair_shop     = data["repair_shop"],
            claimant_name   = data.get("claimant_name", "").strip(),
            ocr_estimate    = float(data["ocr_estimate"]) if data.get("ocr_estimate") else None,
        )

        if mismatch:
            result = _get_identity_mismatch_result(claim_id, data.get("claimant_name", "").strip())
        else:
            async def _run():
                ctx = await _get_nma_context()(claim_id)
                res = await _get_nma_agent()(ctx)
                return ctx, res

            ctx, res = asyncio.run(_run())
            result   = _shape_nma_result(claim_id, res, ctx)

        elapsed = round(time.perf_counter() - t0, 2)
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
        claim_id, mismatch = _inject_claim(
            narrative       = data["narrative"],
            incident_type   = data["incident_type"],
            estimated_loss  = float(data["estimated_loss"]),
            repair_shop     = data["repair_shop"],
            claimant_name   = data.get("claimant_name", "").strip(),
            ocr_estimate    = float(data["ocr_estimate"]) if data.get("ocr_estimate") else None,
        )

        if mismatch:
            short_res = _get_identity_mismatch_result(claim_id, data.get("claimant_name", "").strip())
            mas_result = short_res
            nma_result = short_res
            mas_elapsed = 0.0
            nma_elapsed = 0.0
        else:
            # ── MAS ───────────────────────────────────────────────────────────────
            mas_t0      = time.perf_counter()
            mas_result  = _get_mas()(claim_id)
            mas_elapsed = round(time.perf_counter() - mas_t0, 2)

            # ── NMA ───────────────────────────────────────────────────────────────
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