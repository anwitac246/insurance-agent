"""
ground_truth.py
---------------
Loads the 50-claim test set from MongoDB and derives ground-truth labels
for every evaluation dimension:

  - expected_decision   : "Approved" | "Denied"
  - is_fraud            : bool  (any fraud scenario except "normal")
  - fraud_scenario      : the raw scenario tag from seed_data.py
  - expected_exclusion  : bool  (semantic_exclusion scenario)
  - expected_breach     : bool  (aggregate_breach scenario)

No hardcoded claim IDs or outcomes — everything is derived from
the fraud_scenario tag planted by the seeding script.

Usage
-----
    from evals.ground_truth import load_ground_truth
    gt = load_ground_truth()   # {claim_id: GTRecord}
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()
logger = logging.getLogger(__name__)

# Scenarios that should result in a Denied outcome
_DENY_SCENARIOS = {
    "frequent_claimant",
    "collusion_ring",
    "staged_accident",
    "semantic_exclusion",
    "aggregate_breach",
}

# Scenarios counted as "fraud positive" for precision/recall
_FRAUD_SCENARIOS = {
    "frequent_claimant",
    "collusion_ring",
    "staged_accident",
    "semantic_exclusion",
}


@dataclass
class GTRecord:
    claim_id: str
    customer_id: str
    policy_id: str
    fraud_scenario: str
    incident_type: str
    estimated_loss: float
    narrative: str

    # Derived labels
    expected_decision: str        # "Approved" | "Denied"
    is_fraud: bool                # True → positive class for fraud metrics
    expected_exclusion: bool      # True → policy exclusion should fire
    expected_breach: bool         # True → aggregate limit breach should fire

    # Optional OCR extraction (used by chaos tests)
    ocr_extraction: dict = field(default_factory=dict)


def load_ground_truth(
    mongodb_url: Optional[str] = None,
    db_name: str = "car_insurance_mas",
    collection: str = "Active_Claims",
) -> dict[str, GTRecord]:
    """
    Connect to MongoDB, read every Active_Claim, derive labels,
    and return a mapping {claim_id: GTRecord}.

    Parameters
    ----------
    mongodb_url : str, optional
        Overrides the MONGODB_URL env var (useful for tests).
    db_name     : str
        Database name (default matches seed_data.py).
    collection  : str
        Collection name (default matches seed_data.py).
    """
    url = mongodb_url or os.getenv("MONGODB_URL")
    if not url:
        raise EnvironmentError("MONGODB_URL is not set in the environment or .env file.")

    client = MongoClient(url, serverSelectionTimeoutMS=5_000)
    try:
        # Force a connection check
        client.admin.command("ping")
    except Exception as exc:
        raise ConnectionError(f"Cannot reach MongoDB at {url!r}: {exc}") from exc

    db = client[db_name]
    claims = list(db[collection].find({}, {"_id": 0}))

    if not claims:
        raise ValueError(
            f"Collection '{collection}' in database '{db_name}' is empty. "
            "Run scripts/seed_data.py first."
        )

    gt: dict[str, GTRecord] = {}
    for c in claims:
        scenario: str = c.get("fraud_scenario", "normal")
        cid: str = c["claim_id"]

        gt[cid] = GTRecord(
            claim_id=cid,
            customer_id=c.get("customer_id", ""),
            policy_id=c.get("policy_id", ""),
            fraud_scenario=scenario,
            incident_type=c.get("incident_type", ""),
            estimated_loss=float(c.get("estimated_loss", 0)),
            narrative=c.get("narrative", ""),
            ocr_extraction=c.get("ocr_extraction", {}),
            # Derived labels
            expected_decision="Denied" if scenario in _DENY_SCENARIOS else "Approved",
            is_fraud=scenario in _FRAUD_SCENARIOS,
            expected_exclusion=scenario == "semantic_exclusion",
            expected_breach=scenario == "aggregate_breach",
        )

    logger.info(
        "Loaded %d ground-truth records (%d fraud, %d normal).",
        len(gt),
        sum(1 for r in gt.values() if r.is_fraud),
        sum(1 for r in gt.values() if not r.is_fraud),
    )
    return gt
