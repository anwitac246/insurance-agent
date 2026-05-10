"""
chaos.py
--------
Chaos / robustness testing for the MAS.

v2 fix — THE CORE BUG
----------------------
The previous implementation corrupted a local Python dict and then called
`self._process(claim_id)`, which re-fetches the claim from MongoDB.
The local corruption never reached the pipeline: every chaos run was
identical to the baseline, producing a flat degradation curve of 0.0pp
and making the robustness metric meaningless.

The fix: patch MongoDB directly before calling the pipeline, then restore
the original value in a try/finally block. This ensures the verification
agent (and any other stage that reads ocr_extraction from Active_Claims)
actually receives the corrupted fields.

Patch strategy
--------------
  1. Read the original ocr_extraction from MongoDB.
  2. Build the corrupted version in memory.
  3. $set the corrupted value on the Active_Claims document.
  4. Run self._process(claim_id).
  5. $set the original value back (guaranteed via finally).

Thread safety: each claim is patched and restored sequentially.
Parallel chaos runs would require per-document locks — not implemented here
since the eval loop is single-threaded.

Usage
-----
    from evals.chaos import ChaosHarness
    harness = ChaosHarness(process_fn=process_claim)
    report = harness.run_corruption_sweep(
        ground_truth=gt,
        corruption_levels=[0.0, 0.10, 0.20, 0.30],
        fields_to_corrupt=["PolicyNumber", "ClaimantName", "TotalEstimate"],
    )
"""

from __future__ import annotations

import copy
import logging
import random
import time
from typing import Any, Callable, Optional

from evals.ground_truth import GTRecord
from evals.metrics import decision_accuracy, stp_rate

logger = logging.getLogger(__name__)


def _corrupt_ocr(
    ocr: dict,
    fields: list[str],
    corruption_rate: float,
    rng: random.Random,
) -> dict:
    """
    Return a deep copy of `ocr` with a fraction of the given fields
    set to empty string (simulating an OCR read failure).

    corruption_rate : 0.0 → no corruption, 1.0 → all listed fields blanked.
    """
    mutated = copy.deepcopy(ocr)
    for field in fields:
        if field in mutated and rng.random() < corruption_rate:
            mutated[field] = ""
    return mutated


def _get_mongo_collection():
    """Lazy import so chaos.py can still be imported without a live DB."""
    from src.tools.mongo_client import get_db
    return get_db()["Active_Claims"]


class ChaosHarness:
    """
    Orchestrates robustness experiments.

    Parameters
    ----------
    process_fn : Callable[[str], dict]
        The MAS entry-point — takes a claim_id string, returns a result dict.
    seed : int
        Random seed for reproducible corruption selection.
    """

    def __init__(self, process_fn: Callable[[str], dict], seed: int = 42):
        self._process = process_fn
        self._rng = random.Random(seed)

    # ── Core: patch MongoDB, run pipeline, restore ────────────────────────────

    def run_single_claim(
        self,
        gt_record: GTRecord,
        corruption_rate: float,
        fields: list[str],
        ground_truth: dict[str, GTRecord],
    ) -> dict[str, Any]:
        """
        Corrupt the claim's ocr_extraction in MongoDB, run the pipeline,
        then unconditionally restore the original value.

        This is the only correct way to inject chaos: the verification_agent
        reads ocr_extraction directly from MongoDB, so any in-memory-only
        mutation is silently ignored by the pipeline.
        """
        claim_id = gt_record.claim_id
        original_ocr = copy.deepcopy(gt_record.ocr_extraction)
        corrupted_ocr = _corrupt_ocr(original_ocr, fields, corruption_rate, self._rng)

        corrupted_fields = [
            f for f in fields
            if corrupted_ocr.get(f, "NOT_PRESENT") != original_ocr.get(f, "NOT_PRESENT")
        ]

        # Skip the DB round-trip entirely when corruption_rate == 0
        if corruption_rate == 0.0:
            t0 = time.perf_counter()
            try:
                result = self._process(claim_id)
            except Exception as exc:
                logger.error("Chaos run (0%%) failed for claim %s: %s", claim_id, exc)
                result = {
                    "claim_id": claim_id,
                    "final_decision": {"approved": False, "denial_reason": str(exc)},
                    "errors": [str(exc)],
                }
            elapsed = time.perf_counter() - t0
        else:
            collection = _get_mongo_collection()
            try:
                # 1. Patch MongoDB with corrupted OCR
                collection.update_one(
                    {"claim_id": claim_id},
                    {"$set": {"ocr_extraction": corrupted_ocr}},
                )
                logger.debug(
                    "chaos | claim=%s | patched fields=%s", claim_id, corrupted_fields
                )

                # 2. Run the pipeline against the corrupted data
                t0 = time.perf_counter()
                try:
                    result = self._process(claim_id)
                except Exception as exc:
                    logger.error(
                        "Chaos run (%.0f%%) failed for claim %s: %s",
                        corruption_rate * 100, claim_id, exc,
                    )
                    result = {
                        "claim_id": claim_id,
                        "final_decision": {
                            "approved": False,
                            "denial_reason": str(exc),
                        },
                        "errors": [str(exc)],
                    }
                elapsed = time.perf_counter() - t0

            finally:
                # 3. Restore original OCR — guaranteed even if pipeline raises
                try:
                    collection.update_one(
                        {"claim_id": claim_id},
                        {"$set": {"ocr_extraction": original_ocr}},
                    )
                    logger.debug(
                        "chaos | claim=%s | restored original ocr_extraction", claim_id
                    )
                except Exception as restore_exc:
                    # Log loudly — this is a data integrity issue
                    logger.error(
                        "chaos | FAILED to restore ocr_extraction for claim %s: %s",
                        claim_id, restore_exc,
                    )

        gt = ground_truth.get(claim_id)
        fd = result.get("final_decision") or {}
        predicted = "Approved" if fd.get("approved") else "Denied"
        expected = gt.expected_decision if gt else "Unknown"

        return {
            "claim_id": claim_id,
            "corruption_rate": corruption_rate,
            "corrupted_fields": corrupted_fields,
            "predicted": predicted,
            "expected": expected,
            "correct": predicted == expected,
            "elapsed_s": round(elapsed, 3),
            "errors": result.get("errors", []),
        }

    def run_corruption_sweep(
        self,
        ground_truth: dict[str, GTRecord],
        corruption_levels: Optional[list[float]] = None,
        fields_to_corrupt: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """
        Run every claim at each corruption level and compute accuracy at each level.

        Returns
        -------
        {
            "levels": [0.0, 0.10, 0.20, 0.30],
            "accuracy_by_level": {0.0: 0.94, 0.10: 0.88, ...},
            "stp_by_level": {...},
            "degradation": float,   # accuracy(0%) - accuracy(max%)
            "per_level_details": {...},
        }
        """
        if corruption_levels is None:
            corruption_levels = [0.0, 0.10, 0.20, 0.30]
        if fields_to_corrupt is None:
            fields_to_corrupt = [
                "PolicyNumber", "ClaimantName", "TotalEstimate", "LossDate"
            ]

        accuracy_by_level: dict[float, float] = {}
        stp_by_level: dict[float, float] = {}
        per_level_details: dict[float, list[dict]] = {}

        gt_records = list(ground_truth.values())

        for level in corruption_levels:
            logger.info(
                "Chaos sweep: corruption_rate=%.0f%% (%d claims)",
                level * 100, len(gt_records),
            )
            level_results = []
            level_raw_results = []

            for gt_record in gt_records:
                detail = self.run_single_claim(
                    gt_record, level, fields_to_corrupt, ground_truth
                )
                level_results.append(detail)

                level_raw_results.append({
                    "claim_id": detail["claim_id"],
                    "final_decision": {
                        "approved": detail["predicted"] == "Approved",
                        "denial_reason": "",
                    },
                    "errors": detail["errors"],
                })

            acc = decision_accuracy(level_raw_results, ground_truth)
            stp = stp_rate(level_raw_results)
            accuracy_by_level[level] = acc["accuracy"]
            stp_by_level[level] = stp["stp_rate"]
            per_level_details[level] = level_results

            logger.info(
                "Level %.0f%% → accuracy=%.2f%%  STP=%.2f%%",
                level * 100,
                acc["accuracy"] * 100,
                stp["stp_rate"] * 100,
            )

        baseline_acc = accuracy_by_level.get(0.0, 1.0)
        worst_acc = min(accuracy_by_level.values())
        degradation = round(baseline_acc - worst_acc, 4)

        return {
            "levels": corruption_levels,
            "fields_corrupted": fields_to_corrupt,
            "accuracy_by_level": {
                str(k): round(v, 4) for k, v in accuracy_by_level.items()
            },
            "stp_by_level": {
                str(k): round(v, 4) for k, v in stp_by_level.items()
            },
            "degradation": degradation,
            "per_level_details": {
                str(k): v for k, v in per_level_details.items()
            },
        }