"""
llm_judge.py
------------
LLM-as-Judge utilities using the centralized Groq key rotation manager.

Key fixes vs. previous version:
  - 400 Bad Request errors now fail immediately (non-retryable).
  - Input token limits tightened (policy_text: 1500, facts/steps: 1000 each)
    to stay well within Groq free-tier context limits.
"""

from __future__ import annotations

import logging
import time
from enum import Enum
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.tools.groq_client import get_llm, record_429, record_success

logger = logging.getLogger(__name__)

MAX_RETRIES = 3

# Tightened from 3000/2000 — keeps requests well within Groq context limits
POLICY_TEXT_LIMIT = 1_500
FACTS_LIMIT = 1_000
STEPS_LIMIT = 1_000


# ── Groundedness Judge ─────────────────────────────────────────────────────────

class GroundednessScore(BaseModel):
    score: int = Field(ge=1, le=5, description="1=completely unsupported → 5=fully grounded")
    reasoning: str = Field(description="One-paragraph explanation citing policy text")
    unsupported_claims: list[str] = Field(
        default_factory=list,
        description="Any specific verdicts not supported by the retrieved policy",
    )


_GROUNDEDNESS_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an insurance audit expert. Your task is to verify that a PolicyVerdict "
        "is grounded in the actual retrieved policy text.\n\n"
        "Scoring rubric:\n"
        "  5 — Every claim in the verdict is directly supported by a verbatim or "
        "paraphrased clause in the policy.\n"
        "  4 — Minor inferences but all major claims are supported.\n"
        "  3 — Some claims are supported; others are vague or loosely inferred.\n"
        "  2 — Several unsupported claims; the verdict leans on invented reasoning.\n"
        "  1 — The verdict largely contradicts or ignores the policy text.\n\n"
        "Be strict. Do not give credit for plausible-sounding reasoning that is not "
        "traceable to the provided policy.",
    ),
    (
        "human",
        "=== Retrieved Policy Text ===\n{policy_text}\n\n"
        "=== PolicyVerdict produced by the agent ===\n{verdict_text}\n\n"
        "Score this verdict for groundedness.",
    ),
])


class GroundednessJudge:
    def _invoke(self, inputs: dict) -> GroundednessScore:
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
                    raise
                if "429" in exc_str or "rate limit" in exc_str or "rate_limit" in exc_str:
                    logger.warning(
                        "groundedness_judge | 429 (attempt %d)", attempt + 1
                    )
                    record_429()
                    time.sleep(2 ** attempt)
                else:
                    logger.warning(
                        "groundedness_judge | error (attempt %d): %s", attempt + 1, exc
                    )
                    time.sleep(2 ** attempt)
        raise last_exc

    def score(self, policy_text: str, verdict: dict) -> GroundednessScore:
        verdict_text = (
            f"incident_covered={verdict.get('incident_covered')}, "
            f"exclusion_triggered={verdict.get('exclusion_triggered')}, "
            f"exclusion_reason={verdict.get('exclusion_reason', '')!r}, "
            f"reasoning={verdict.get('coverage_reasoning', '')}"
        )
        return self._invoke({
            # Tightened limit to stay within free-tier context window
            "policy_text": policy_text[:POLICY_TEXT_LIMIT],
            "verdict_text": verdict_text,
        })


# ── Hallucination Judge ────────────────────────────────────────────────────────

class SupportLabel(str, Enum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    UNCERTAIN = "uncertain"


class StepJudgment(BaseModel):
    step: str
    label: SupportLabel
    reason: str = ""


class HallucinationReport(BaseModel):
    judgments: list[StepJudgment]
    hallucination_rate: float = Field(
        description="Fraction of steps labeled 'unsupported'"
    )


_HALLUCINATION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a factual accuracy auditor. For each reasoning step, classify it as:\n"
        "  supported   — directly derivable from the provided facts/data\n"
        "  unsupported — makes a claim not traceable to the provided facts\n"
        "  uncertain   — plausible but cannot be verified from the facts alone\n\n"
        "Be precise. Each step is one sentence or short clause.",
    ),
    (
        "human",
        "=== Source Facts (claim data, policy, fraud signals) ===\n{facts}\n\n"
        "=== Agent Reasoning Steps ===\n{steps}\n\n"
        "Classify each step. Then compute hallucination_rate = "
        "unsupported_count / total_count.",
    ),
])


class HallucinationJudge:
    def _invoke(self, inputs: dict) -> HallucinationReport:
        last_exc = None
        for attempt in range(MAX_RETRIES):
            try:
                llm = get_llm()
                chain = _HALLUCINATION_PROMPT | llm.with_structured_output(HallucinationReport)
                result = chain.invoke(inputs)
                record_success()
                return result
            except Exception as exc:
                last_exc = exc
                exc_str = str(exc).lower()
                if "400" in exc_str or "bad request" in exc_str:
                    logger.error(
                        "hallucination_judge | 400 Bad Request (non-retryable): %s", exc
                    )
                    raise
                if "429" in exc_str or "rate limit" in exc_str or "rate_limit" in exc_str:
                    logger.warning(
                        "hallucination_judge | 429 (attempt %d)", attempt + 1
                    )
                    record_429()
                    time.sleep(2 ** attempt)
                else:
                    logger.warning(
                        "hallucination_judge | error (attempt %d): %s", attempt + 1, exc
                    )
                    time.sleep(2 ** attempt)
        raise last_exc

    def evaluate(self, facts: str, reasoning: str) -> HallucinationReport:
        raw_steps = [
            s.strip()
            for s in reasoning.replace("\n", ". ").split(".")
            if len(s.strip()) > 20
        ]
        steps_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(raw_steps))

        return self._invoke({
            # Tightened limits to stay within free-tier context window
            "facts": facts[:FACTS_LIMIT],
            "steps": steps_text[:STEPS_LIMIT],
        })