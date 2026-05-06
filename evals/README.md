# Evaluation Pipeline — Car Insurance MAS

A production-grade, dynamic evaluation harness that measures correctness,
reliability, robustness, and reasoning quality without hardcoding any values.

---

## Quick Start

```bash
# Full evaluation (K=3 consistency runs, chaos sweep, LLM judge)
python -m evals.run_evals

# Fast run (single pass, no chaos, no LLM judge)
python -m evals.run_evals --k 1 --no-chaos --no-llm-judge

# Evaluate specific claims
python -m evals.run_evals --claim-ids <id1> <id2> --k 1 --no-chaos

# Evaluate first 10 claims only (for quick smoke testing)
python -m evals.run_evals --claims 10 --k 1 --no-chaos --no-llm-judge
```

Results are saved to `evals/results/`:
- `eval_report_<timestamp>.json`  — full metric report
- `eval_chart_<timestamp>.png`    — 6-panel comparison dashboard

---

## Architecture

```
evals/
├── __init__.py
├── run_evals.py       ← Top-level orchestrator (entry point)
├── ground_truth.py    ← Loads labels from MongoDB (no hardcoding)
├── metrics.py         ← Pure-Python metric calculations
├── llm_judge.py       ← Groundedness & Hallucination scoring (Groq only)
├── chaos.py           ← OCR field corruption / robustness testing
├── reporter.py        ← JSON report + Matplotlib dashboard
└── results/           ← Generated outputs (git-ignored)
```

---

## Metrics

### 1. Decision Accuracy
Compares `final_decision.approved` against the ground-truth label derived
from each claim's `fraud_scenario` tag:

| Scenario           | Expected Decision |
|--------------------|-------------------|
| normal             | Approved          |
| frequent_claimant  | Denied            |
| collusion_ring     | Denied            |
| staged_accident    | Denied            |
| semantic_exclusion | Denied            |
| aggregate_breach   | Denied            |

### 2. Fraud Precision / Recall / F1
Binary classification: the fraud agent's `risk_score == "High"` is the
predicted positive. Ground truth positive = any non-normal scenario.

### 3. Straight-Through Processing (STP) Rate
Percentage of claims that reached the `decision` node without being
short-circuited by the `failure_node`. A healthy MAS should have >90% STP.

### 4. Groundedness Score (LLM-as-Judge, 1–5)
Groq judges whether the `PolicyVerdict.coverage_reasoning` is supported by
the actual retrieved policy text. Scored 1 (unsupported) → 5 (fully grounded).

### 5. Hallucination Rate
Each sentence in `final_decision.step_by_step_reasoning` is classified as:
- `supported` — derivable from claim data
- `unsupported` — fabricated / not traceable to facts
- `uncertain` — plausible but unverifiable

`hallucination_rate = unsupported_count / total_steps`

### 6. Consistency (K runs)
Each claim is processed K independent times. Consistency per claim:
`1 - (differing_decisions / K)`

### 7. Step Completeness
Required checkpoints per agent are defined in `metrics.REQUIRED_STEPS`.
Completeness = executed checkpoints / required checkpoints.

### 8. Latency & Token Efficiency
Wall-clock time per claim. Token counts per agent (requires LangChain
callback wiring — returns 0 if not wired up, non-fatal).

### 9. Chaos / Robustness
Accuracy is measured at 0%, 10%, 20%, and 30% OCR field corruption rates.
`degradation = accuracy(0%) - accuracy(max%)`

---

## Ground Truth Labels

Labels are derived **dynamically** from the `fraud_scenario` field planted by
`scripts/seed_data.py`. There are zero hardcoded claim IDs or outcomes.

If you re-seed the database with different distributions, the evaluation
pipeline automatically adapts.

---

## LLM Provider

All LLM calls (agents + evaluation judges) use **Groq** exclusively.
No Anthropic API key is required anywhere in this project.

Model used: `llama-3.1-8b-instant`

---

## Adding Custom Metrics

1. Add a pure function to `metrics.py` that takes `results` + `ground_truth`.
2. Call it in `run_evals.py` and add the result to the `report` dict.
3. Optionally add a panel to `reporter.py`.
