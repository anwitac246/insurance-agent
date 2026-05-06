"""
llm_judge.py
------------
LLM-as-Judge utilities powered exclusively by Groq (llama-3.3-70b-versatile).
No Anthropic API key is required anywhere in this module.

Two judges:
  1. GroundednessJudge  — scores PolicyVerdict against retrieved policy text (1-5)
  2. HallucinationJudge — classifies individual reasoning steps as supported / unsupported

Both judges use structured outputs (Pydantic) so results can be parsed
deterministically downstream.
"""

from __future__ import annotations

import logging
import os
import time
from enum import Enum
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

load_dotenv()
logger = logging.getLogger(__name__)

GROQ_MODEL = "llama-3.3-70b-versatile"
MAX_RETRIES = 3


# ── Lazy singleton ─────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _get_llm() -> ChatGroq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY is not set in the environment or .env file.")
    return ChatGroq(model=GROQ_MODEL, temperature=0, request_timeout=45)


# ── Retry decorator ────────────────────────────────────────────────────────────
def _retry():
    return retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(min=2, max=10),
        reraise=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 1. Groundedness Judge
# ══════════════════════════════════════════════════════════════════════════════

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
    """Score how well a PolicyVerdict is supported by retrieved policy text."""

    def __init__(self):
        self._chain = _GROUNDEDNESS_PROMPT | _get_llm().with_structured_output(GroundednessScore)

    @_retry()
    def score(self, policy_text: str, verdict: dict) -> GroundednessScore:
        verdict_text = (
            f"incident_covered={verdict.get('incident_covered')}, "
            f"exclusion_triggered={verdict.get('exclusion_triggered')}, "
            f"exclusion_reason={verdict.get('exclusion_reason', '')!r}, "
            f"reasoning={verdict.get('coverage_reasoning', '')}"
        )
        return self._chain.invoke({
            "policy_text": policy_text[:3000],  # guard against context overflow
            "verdict_text": verdict_text,
        })


# ══════════════════════════════════════════════════════════════════════════════
# 2. Hallucination Judge
# ══════════════════════════════════════════════════════════════════════════════

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
    """Classify each reasoning step as supported, unsupported, or uncertain."""

    def __init__(self):
        self._chain = _HALLUCINATION_PROMPT | _get_llm().with_structured_output(
            HallucinationReport
        )

    @_retry()
    def evaluate(self, facts: str, reasoning: str) -> HallucinationReport:
        # Split reasoning into individual steps by sentence / newline
        raw_steps = [
            s.strip()
            for s in reasoning.replace("\n", ". ").split(".")
            if len(s.strip()) > 20
        ]
        steps_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(raw_steps))

        return self._chain.invoke({
            "facts": facts[:2000],
            "steps": steps_text[:2000],
        })
