# Single-Agent System (NMA) Benchmark Baseline

## Overview
This directory contains the **Single-Agent System (Non-Multi-Agent)** architecture for the insurance claim adjudication project. 

It was built strictly as a baseline benchmarking comparison against the robust LangGraph Multi-Agent System (MAS) located in the root repository. By evaluating both systems against the exact same 50-claim MongoDB dataset, we can empirically measure the trade-offs between monolithic prompting and graph-based multi-agent routing.

---

## The Goal
The primary objective of the NMA is to test the limits of modern Large Language Models (specifically `llama-3.3-70b-versatile` on Groq) when forced to handle highly complex, multi-step reasoning tasks in a single pass. 

We want to answer: *Does breaking a problem into specialized agents (MAS) actually improve accuracy, or can a single prompt with sufficient context perform just as well?*

---

## Tech Stack
* **LLM Provider**: Groq (`llama-3.3-70b-versatile` for high-context capacity)
* **Context Retrieval**: PyMongo (MongoDB) & Pinecone (Vector database)
* **Structured Output Validation**: Pydantic v2
* **Evaluation Framework**: Custom Python harness mirroring the MAS `evals.metrics` module.

---

## Implementation Strategy

Unlike the Multi-Agent System which splits the claim processing pipeline into discrete specialized nodes (Verification, Fraud, Policy, Decision), this system relies entirely on a monolithic approach:

1. **Massive Context Aggregation (`context_fetcher.py`)**: 
   Instead of agents conditionally querying data as needed, the NMA script performs a single, massive data fetch up-front. It pulls the Customer Profile, the Active Claim data, the historical Claim History records, and the Pinecone Policy Exclusions all at once and bundles them into a single context object.
   
2. **Deterministic Pre-Computation**: 
   To prevent hallucination on rigid mathematical logic, the script pre-computes certain signals (e.g., "Has the customer breached the frequent claimant threshold of 3?"). These signals are injected directly into the prompt alongside the raw data.

3. **Monolithic Prompting (`nma_agent.py`)**: 
   The massive context payload is injected into a single, highly complex system prompt. The LLM is forced via a strict Pydantic schema to output a single JSON object that simultaneously assigns a fraud risk score, determines policy coverage, applies exclusions, and calculates the final financial payout.

4. **Deterministic Overrides**: 
   Post-LLM execution, the Python script applies hard deterministic overrides. If the mathematical "remaining limit" is 0, the script overrides the LLM and forces the payout to $0.00, ensuring absolute financial safety.

---

## Usage

*Important: You must run these commands from the root `insurance-agent` directory so the scripts can properly access the shared `.env` variables and database configurations.*

### Manual Claim Testing
To test the single agent on an interactive prompt where you supply a specific Claim ID from your MongoDB:
```bash
python insurance-agent-NMA/nma_src/main.py
```

### Run the Evaluation Benchmark
To run the Single-Agent System against the 50-claim Ground Truth dataset and calculate accuracy, latency, and token efficiency metrics:
```bash
python insurance-agent-NMA/evals/run_evals_nma.py
```

*The resulting JSON report will be saved to `evals/results/` for easy side-by-side comparison with the MAS reports.*
