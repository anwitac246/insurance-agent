"""
llm_judge.py
------------
LLM-as-Judge utilities using the centralized Groq key rotation manager.

Key fixes vs. previous version:
  - HallucinationReport no longer uses .with_structured_output() with a
    nested Pydantic schema (list[StepJudgment]) — this was the primary cause
    of 400 Bad Request errors on Groq free tier. Plain text parsing is used
    instead, which is far more reliable for free-tier context limits.
  - GroundednessScore still uses structured output but with a simplified
    schema (score + reasoning only, no list field) to avoid 400s.
  - Input token limits tightened further (policy_text: 800, facts/steps: 600).
  - Both judges now fall back gracefully on 400s instead of raising.

FIX vs previous version:
  - _split_reasoning_steps() now correctly handles the 3-sentence numbered
    format produced by the updated decision_agent. The old splitter used
    reasoning.split(".") which fragmented numbered sentences like
    "1. Policy covered. 2. No fraud." into broken sub-fragments, causing
    the hallucination judge to see 0-1 scoreable steps per claim and produce
    artificially high hallucination rates (often 1.0 for a single-step claim).
  - New splitter tries numbered-step regex first (matching "1.", "2.", "3.")
    then falls back to sentence splitting — preserving the 3-step structure
    that maps directly to verifiable facts in the expanded facts string.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.tools.llm_client import get_llm, record_429, record_success

logger = logging.getLogger(__name__)

MAX_RETRIES = 3

# Tightened further — keeps requests well within Groq free-tier context limits
POLICY_TEXT_LIMIT = 800
FACTS_LIMIT = 900   # increased slightly to accommodate richer facts string
STEPS_LIMIT = 600


# ── Groundedness Judge ─────────────────────────────────────────────────────────
# Simplified schema: removed list[str] field which caused 400s with complex JSON schema

class GroundednessScore(BaseModel):
    score: int = Field(ge=1, le=5, description="1=completely unsupported → 5=fully grounded")
    reasoning: str = Field(description="One-paragraph explanation")


_GROUNDEDNESS_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an insurance audit expert. Score whether a PolicyVerdict is grounded "
        "in the retrieved policy text.\n\n"
        "Scoring rubric:\n"
        "  5 — Every claim directly supported by the policy text.\n"
        "  4 — Minor inferences, major claims supported.\n"
        "  3 — Some supported, some vague.\n"
        "  2 — Several unsupported claims.\n"
        "  1 — Largely contradicts or ignores the policy.\n\n"
        "Be strict. Only credit reasoning traceable to the provided policy.",
    ),
    (
        "human",
        "=== Policy Text ===\n{policy_text}\n\n"
        "=== PolicyVerdict ===\n{verdict_text}\n\n"
        "Score this verdict for groundedness.",
    ),
])


class GroundednessJudge:
    def _invoke(self, inputs: dict) -> Optional[GroundednessScore]:
        last_exc = None
        for attempt in range(MAX_RETRIES):
            try:
                llm = get_llm()
                chain = _GROUNDEDNESS_PROMPT | llm.with_structured_output(GroundednessScore)
                result = chain.invoke(inputs)
                record_success()
                return result
            except Exception as exc:
                last_exc = exc
                exc_str = str(exc).lower()
                if "400" in exc_str or "bad request" in exc_str:
                    logger.error(
                        "groundedness_judge | 400 Bad Request (non-retryable): %s", exc
                    )
                    return None  # Fail gracefully — don't crash the eval run
                if "429" in exc_str or "rate limit" in exc_str or "rate_limit" in exc_str:
                    logger.warning("groundedness_judge | 429 (attempt %d)", attempt + 1)
                    record_429()
                    time.sleep(2 ** attempt)
                else:
                    logger.warning(
                        "groundedness_judge | error (attempt %d): %s", attempt + 1, exc
                    )
                    time.sleep(2 ** attempt)
        logger.error("groundedness_judge | all retries failed: %s", last_exc)
        return None

    def score(self, policy_text: str, verdict: dict) -> Optional[GroundednessScore]:
        verdict_text = (
            f"incident_covered={verdict.get('incident_covered')}, "
            f"exclusion_triggered={verdict.get('exclusion_triggered')}, "
            f"exclusion_reason={verdict.get('exclusion_reason', '')!r}, "
            f"reasoning={verdict.get('coverage_reasoning', '')}"
        )
        # Hard-truncate verdict_text too so total input stays small
        return self._invoke({
            "policy_text": policy_text[:POLICY_TEXT_LIMIT],
            "verdict_text": verdict_text[:400],
        })


# ── Hallucination Judge ────────────────────────────────────────────────────────
# Use plain-text LLM output instead of structured output — avoids 400 Bad Request
# errors caused by complex nested JSON schemas (list[StepJudgment]) on Groq.

class HallucinationReport:
    """Simple result container — not a Pydantic model to avoid schema issues."""
    def __init__(self, hallucination_rate: float, supported: int, unsupported: int, uncertain: int):
        self.hallucination_rate = hallucination_rate
        self.supported = supported
        self.unsupported = unsupported
        self.uncertain = uncertain


_HALLUCINATION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a factual accuracy auditor. For each numbered reasoning step, "
        "output one line: '<number>: supported|unsupported|uncertain'\n"
        "  supported   — directly derivable from the provided facts\n"
        "  unsupported — makes a claim not traceable to the facts\n"
        "  uncertain   — plausible but unverifiable from the facts\n\n"
        "After all steps, output a final line: 'hallucination_rate: <0.0-1.0>'\n"
        "No other text.",
    ),
    (
        "human",
        "=== Source Facts ===\n{facts}\n\n"
        "=== Reasoning Steps ===\n{steps}\n\n"
        "Classify each step, then give hallucination_rate.",
    ),
])

_LABEL_RE = re.compile(r"^\d+:\s*(supported|unsupported|uncertain)", re.IGNORECASE)
_RATE_RE = re.compile(r"hallucination_rate:\s*([0-9.]+)", re.IGNORECASE)

# FIX: Numbered step pattern — matches "1.", "2.", "3." at line start, after newline, or after space.
# The LLM often outputs all 3 steps on a single line separated by spaces.
_NUMBERED_STEP_RE = re.compile(r"(?:^|\s+)\d+\.\s+")


def _split_reasoning_steps(reasoning: str) -> list[str]:
    """
    Split decision agent reasoning into scoreable atomic steps.

    Strategy:
      1. Try numbered step pattern first: "1. ...", "2. ...", "3. ..."
         This is the format produced by the updated decision_agent prompt.
         If we get >= 2 non-empty segments, use them directly.
      2. Fallback: sentence splitting on [.!?] — handles legacy single-sentence
         reasoning from older pipeline runs stored in eval history.

    Returns a list of step strings (max 8 for token budget).
    """
    # Strategy 1: numbered steps
    parts = _NUMBERED_STEP_RE.split(reasoning.strip())
    steps = [p.strip().rstrip(".") for p in parts if len(p.strip()) > 15]
    if len(steps) >= 2:
        return steps[:8]

    # Strategy 2: sentence split fallback
    sentences = re.split(r"[.!?]\s+", reasoning)
    return [s.strip() for s in sentences if len(s.strip()) > 15][:8]


def _parse_hallucination_text(text: str) -> HallucinationReport:
    """Parse plain-text step labels and extract hallucination rate."""
    supported = unsupported = uncertain = 0
    rate = None

    for line in text.splitlines():
        line = line.strip()
        m = _LABEL_RE.match(line)
        if m:
            label = m.group(1).lower()
            if label == "supported":
                supported += 1
            elif label == "unsupported":
                unsupported += 1
            else:
                uncertain += 1
        r = _RATE_RE.search(line)
        if r:
            try:
                rate = float(r.group(1))
            except ValueError:
                pass

    total = supported + unsupported + uncertain
    if rate is None:
        rate = unsupported / total if total > 0 else 0.0

    return HallucinationReport(
        hallucination_rate=round(rate, 4),
        supported=supported,
        unsupported=unsupported,
        uncertain=uncertain,
    )


class HallucinationJudge:
    def _invoke(self, inputs: dict) -> Optional[HallucinationReport]:
        last_exc = None
        for attempt in range(MAX_RETRIES):
            try:
                llm = get_llm()
                # Plain text output — no .with_structured_output() to avoid 400s
                chain = _HALLUCINATION_PROMPT | llm
                result = chain.invoke(inputs)
                record_success()
                return _parse_hallucination_text(result.content)
            except Exception as exc:
                last_exc = exc
                exc_str = str(exc).lower()
                if "400" in exc_str or "bad request" in exc_str:
                    logger.error(
                        "hallucination_judge | 400 Bad Request (non-retryable): %s", exc
                    )
                    return None  # Fail gracefully
                if "429" in exc_str or "rate limit" in exc_str or "rate_limit" in exc_str:
                    logger.warning("hallucination_judge | 429 (attempt %d)", attempt + 1)
                    record_429()
                    time.sleep(2 ** attempt)
                else:
                    logger.warning(
                        "hallucination_judge | error (attempt %d): %s", attempt + 1, exc
                    )
                    time.sleep(2 ** attempt)
        logger.error("hallucination_judge | all retries failed: %s", last_exc)
        return None

    def evaluate(self, facts: str, reasoning: str) -> Optional[HallucinationReport]:
        # FIX: Use the new structured step splitter instead of split(".")
        raw_steps = _split_reasoning_steps(reasoning)
        steps_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(raw_steps))

        return self._invoke({
            "facts": facts[:FACTS_LIMIT],
            "steps": steps_text[:STEPS_LIMIT],
        })