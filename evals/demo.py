"""
demo.py
-------
Interactive demo tool for the Car Insurance MAS.

Two modes:

  1. By claim ID (from the seeded database):
       python demo.py --id <claim_id>

  2. Manual input (inject your own scenario — no DB seed required):
       python demo.py --manual

  3. List available claim IDs grouped by fraud scenario:
       python demo.py --list

  4. Run a preset scenario by name (great for demos):
       python demo.py --preset staged_accident
       python demo.py --preset collusion_ring
       python demo.py --preset normal
       python demo.py --preset fraud_high_risk
       python demo.py --preset semantic_exclusion

Usage examples
--------------
  # Quick demo with a preset
  python demo.py --preset staged_accident

  # Test a real seeded claim
  python demo.py --id abc123-...

  # Fully custom input
  python demo.py --manual
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import uuid
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.WARNING,   # suppress internal logs during demo for clean output
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# ── Preset scenarios for demo use ─────────────────────────────────────────────

PRESETS = {
    "normal": {
        "description": "Clean rear-end collision — should be APPROVED",
        "incident_type": "Rear-end",
        "narrative": (
            "While stopped at a red light on Main Street, the claimant's vehicle "
            "was struck from behind by a white sedan travelling at approximately "
            "30 mph. The driver of the other vehicle admitted fault at the scene. "
            "A police report was filed. Both airbags deployed and the rear bumper, "
            "trunk lid, and exhaust system require replacement."
        ),
        "estimated_loss": 8500.00,
        "repair_shop": "Sunrise Auto Body",
        "ocr_estimate": None,   # same as estimated_loss
    },
    "staged_accident": {
        "description": "Catastrophic narrative but tiny repair bill — STAGING signal",
        "incident_type": "Rear-end",
        "narrative": (
            "A massive multi-vehicle pileup occurred on the interstate involving "
            "at least fourteen vehicles including three semi-trucks. Multiple lanes "
            "were blocked for six hours. Emergency services, fire department, and "
            "two ambulances were on scene. The claimant's vehicle was caught in the "
            "centre of the pile-up and sustained extensive structural damage."
        ),
        "estimated_loss": 18000.00,
        "repair_shop": "Highway Collision Repairs",
        "ocr_estimate": 200.0,   # tiny OCR estimate = staging signal
    },
    "collusion_ring": {
        "description": "Flagged repair shop used by a known collusion network — FRAUD signal",
        "incident_type": "Vandalism",
        "narrative": (
            "Claimant returned to the car park to find the driver-side door heavily "
            "dented and the wing mirror broken off. No witnesses were present and "
            "no CCTV footage is available from the location."
        ),
        "estimated_loss": 3200.00,
        "repair_shop": "Apex AutoBody & Collision",  # the flagged shop
        "ocr_estimate": None,
    },
    "semantic_exclusion": {
        "description": "Track-day incident — racing exclusion should fire",
        "incident_type": "Total Loss",
        "narrative": (
            "The claimant was driving at high speed on a closed circuit track "
            "during a private performance event. The vehicle lost traction on a "
            "hairpin corner at approximately 110 mph and struck the armco barrier. "
            "The chassis is bent and the engine bay sustained fire damage."
        ),
        "estimated_loss": 24000.00,
        "repair_shop": "Motorsport Engineering Ltd",
        "ocr_estimate": None,
    },
    "fraud_high_risk": {
        "description": "High-value claim from a High Risk customer with bad history",
        "incident_type": "Theft",
        "narrative": (
            "Claimant states the vehicle was stolen overnight from outside their "
            "residence. No alarm was triggered despite an active security system. "
            "A police report was filed the following morning."
        ),
        "estimated_loss": 22000.00,
        "repair_shop": "N/A - Total Loss",
        "ocr_estimate": None,
        "force_history": "bad",   # signal to inject bad history when building claim
    },
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _print_banner(title: str) -> None:
    width = 65
    print(f"\n{'═' * width}")
    print(f"  {title}")
    print(f"{'═' * width}")


def _print_section(title: str, data) -> None:
    print(f"\n{'─' * 65}")
    print(f"  {title}")
    print(f"{'─' * 65}")
    if isinstance(data, dict):
        print(json.dumps(data, indent=2, default=str))
    elif isinstance(data, list):
        if data:
            for item in data:
                print(f"  • {item}")
        else:
            print("  (none)")
    else:
        print(f"  {data}")


def _print_result(result: dict) -> None:
    decision = result.get("final_decision") or {}
    approved = decision.get("approved", False)
    payout = result.get("final_payout", 0.0)

    _print_section("SANITIZED DATA", result.get("sanitized_data", {}))

    pv = result.get("policy_verdict")
    if pv:
        _print_section("POLICY VERDICT", {
            "incident_covered": pv.get("incident_covered"),
            "exclusion_triggered": pv.get("exclusion_triggered"),
            "exclusion_reason": pv.get("exclusion_reason") or "—",
            "remaining_limit": f"${pv.get('remaining_limit', 0):,.2f}",
            "coverage_reasoning": pv.get("coverage_reasoning", ""),
        })

    fr = result.get("fraud_report")
    if fr:
        _print_section("FRAUD REPORT", {
            "risk_score": fr.get("risk_score"),
            "frequent_claims_flag": fr.get("frequent_claims_flag"),
            "collusion_flag": fr.get("collusion_flag"),
            "staging_flag": fr.get("staging_flag"),
            "anomalies": fr.get("anomalies", []),
        })

    errors = result.get("errors", [])
    if errors:
        _print_section("ERRORS / FLAGS", errors)

    width = 65
    status = "✓  APPROVED" if approved else "✗  DENIED"
    print(f"\n{'═' * width}")
    print(f"  FINAL DECISION : {status}")
    print(f"  FINAL PAYOUT   : ${payout:,.2f}")
    print(f"{'═' * width}")
    reasoning = decision.get("step_by_step_reasoning", "")
    if reasoning:
        print(f"\nReasoning:\n{reasoning}\n")


# ── DB helpers ─────────────────────────────────────────────────────────────────

def _list_claims() -> None:
    """Print all seeded claim IDs grouped by fraud scenario."""
    try:
        from src.tools.mongo_client import get_db
        db = get_db()
        claims = list(db["Active_Claims"].find({}, {"_id": 0, "claim_id": 1, "fraud_scenario": 1, "incident_type": 1, "estimated_loss": 1}))
    except Exception as exc:
        print(f"ERROR: Could not connect to MongoDB: {exc}")
        sys.exit(1)

    if not claims:
        print("No claims found. Run: python scripts/seed_data.py")
        return

    from collections import defaultdict
    grouped: dict[str, list] = defaultdict(list)
    for c in claims:
        grouped[c["fraud_scenario"]].append(c)

    _print_banner(f"Seeded Claims ({len(claims)} total)")
    for scenario in sorted(grouped):
        print(f"\n  [{scenario.upper()}]")
        for c in grouped[scenario]:
            print(
                f"    {c['claim_id']}  "
                f"{c['incident_type']:<14}  "
                f"${c['estimated_loss']:>10,.2f}"
            )
    print()


def _inject_claim(
    narrative: str,
    incident_type: str,
    estimated_loss: float,
    repair_shop: str,
    claimant_name: str,
    ocr_estimate: float | None = None,
    policy_id: str | None = None,
    customer_id: str | None = None,
) -> str:
    """
    Insert a temporary claim into MongoDB and return its claim_id.
    If policy_id/customer_id are not provided, borrows the first existing
    customer from the DB so the policy lookup works end-to-end.
    """
    from src.tools.mongo_client import get_db
    db = get_db()

    # Borrow an existing customer so Pinecone policy lookup works
    if not customer_id or not policy_id:
        existing = db["Customer_Profiles"].find_one({})
        if not existing:
            print("ERROR: No customers in DB. Run: python scripts/seed_data.py")
            sys.exit(1)
        customer_id = existing["customer_id"]
        policy_id = existing["policy_id"]
        claimant_name = existing.get("full_name", claimant_name)
        print(f"  (Borrowing customer '{claimant_name}' / policy {policy_id[:8].upper()}…)")

    claim_id = str(uuid.uuid4())
    loss_date = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d")

    claim = {
        "claim_id": claim_id,
        "policy_id": policy_id,
        "customer_id": customer_id,
        "incident_type": incident_type,
        "incident_date": loss_date,
        "narrative": narrative,
        "ocr_extraction": {
            "PolicyNumber": policy_id[:8].upper(),
            "ClaimantName": claimant_name,
            "LossDate": loss_date,
            "RepairShopName": repair_shop,
            "TotalEstimate": ocr_estimate if ocr_estimate is not None else estimated_loss,
        },
        "estimated_loss": estimated_loss,
        "fraud_scenario": "manual_demo",
    }
    db["Active_Claims"].insert_one(claim)
    return claim_id


def _cleanup_claim(claim_id: str) -> None:
    """Remove the temporary demo claim from the DB."""
    try:
        from src.tools.mongo_client import get_db
        db = get_db()
        db["Active_Claims"].delete_one({"claim_id": claim_id})
    except Exception:
        pass


# ── Mode: by claim ID ──────────────────────────────────────────────────────────

def run_by_id(claim_id: str) -> None:
    from src.main import process_claim
    _print_banner(f"Processing Claim: {claim_id}")
    print("  Running pipeline… (this takes ~15-30s on Groq free tier)\n")
    result = process_claim(claim_id)
    _print_result(result)


# ── Mode: preset scenario ──────────────────────────────────────────────────────

def run_preset(name: str) -> None:
    if name not in PRESETS:
        print(f"Unknown preset '{name}'. Available: {', '.join(PRESETS)}")
        sys.exit(1)

    preset = PRESETS[name]
    _print_banner(f"Demo Preset: {name.upper()}")
    print(f"  Scenario : {preset['description']}")
    print(f"  Narrative: {preset['narrative'][:120]}…")
    print(f"  Loss     : ${preset['estimated_loss']:,.2f}")
    if preset.get("ocr_estimate"):
        print(f"  OCR Est  : ${preset['ocr_estimate']:,.2f}  ← mismatch signal")
    print(f"  Shop     : {preset['repair_shop']}")
    print("\n  Injecting claim into DB and running pipeline…\n")

    from src.main import process_claim

    claim_id = _inject_claim(
        narrative=preset["narrative"],
        incident_type=preset["incident_type"],
        estimated_loss=preset["estimated_loss"],
        repair_shop=preset["repair_shop"],
        claimant_name="Demo User",
        ocr_estimate=preset.get("ocr_estimate"),
    )

    try:
        result = process_claim(claim_id)
        _print_result(result)
    finally:
        _cleanup_claim(claim_id)
        print(f"  (Temporary claim {claim_id[:8]}… removed from DB)")


# ── Mode: fully manual ─────────────────────────────────────────────────────────

def run_manual() -> None:
    _print_banner("Manual Claim Input")
    print("  Enter claim details. Press Enter to accept defaults.\n")

    incident_types = ["Rear-end", "Theft", "Vandalism", "Hit and Run", "Total Loss"]
    print(f"  Incident types: {', '.join(incident_types)}")
    incident_type = input("  Incident type [Rear-end]: ").strip() or "Rear-end"

    print("\n  Paste the claim narrative (single line, press Enter when done):")
    narrative = input("  Narrative: ").strip()
    if not narrative:
        narrative = (
            "The claimant's vehicle was struck from behind at a traffic light. "
            "The other driver admitted fault. Police report filed."
        )
        print(f"  Using default: {narrative}")

    loss_str = input("\n  Estimated loss amount [$5000]: ").strip() or "5000"
    try:
        estimated_loss = float(loss_str.replace(",", "").replace("$", ""))
    except ValueError:
        estimated_loss = 5000.0

    repair_shop = input("  Repair shop name [City Auto Body]: ").strip() or "City Auto Body"

    ocr_str = input(f"  OCR repair estimate (leave blank to use ${estimated_loss:,.2f}): ").strip()
    if ocr_str:
        try:
            ocr_estimate = float(ocr_str.replace(",", "").replace("$", ""))
        except ValueError:
            ocr_estimate = None
    else:
        ocr_estimate = None

    print("\n  Injecting claim and running pipeline…\n")

    from src.main import process_claim

    claim_id = _inject_claim(
        narrative=narrative,
        incident_type=incident_type,
        estimated_loss=estimated_loss,
        repair_shop=repair_shop,
        claimant_name="Demo User",
        ocr_estimate=ocr_estimate,
    )

    try:
        result = process_claim(claim_id)
        _print_result(result)
    finally:
        _cleanup_claim(claim_id)
        print(f"  (Temporary claim {claim_id[:8]}… removed from DB)")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Car Insurance MAS — Interactive Demo Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo.py --list
  python demo.py --preset staged_accident
  python demo.py --preset normal
  python demo.py --preset collusion_ring
  python demo.py --preset semantic_exclusion
  python demo.py --preset fraud_high_risk
  python demo.py --id <claim_id_from_--list>
  python demo.py --manual
        """,
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true", help="List all seeded claim IDs")
    group.add_argument("--id", metavar="CLAIM_ID", help="Process a specific claim by ID")
    group.add_argument(
        "--preset", metavar="NAME",
        choices=list(PRESETS.keys()),
        help=f"Run a preset demo scenario: {', '.join(PRESETS)}",
    )
    group.add_argument("--manual", action="store_true", help="Enter claim details interactively")
    args = p.parse_args()

    if args.list:
        _list_claims()
    elif args.id:
        run_by_id(args.id)
    elif args.preset:
        run_preset(args.preset)
    elif args.manual:
        run_manual()


if __name__ == "__main__":
    main()