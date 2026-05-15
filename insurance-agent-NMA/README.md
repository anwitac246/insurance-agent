# Single-Agent System Benchmark Baseline (`insurance-agent-NMA/`)

This directory contains the **Single-Agent System (Non-Multi-Agent)** architecture. It was built strictly as a baseline benchmarking comparison against the robust LangGraph Multi-Agent System (MAS) located in the root repository. 

## Implementation Strategy
Unlike the MAS, which splits the claim processing pipeline into discrete specialized nodes, this system relies entirely on a monolithic approach to test the cognitive load limits of the LLM.

1. **Massive Context Aggregation (`nma_src/tools/context_fetcher.py`)**
   Instead of conditionally querying data as needed, this script performs a single massive data fetch upfront. It pulls the Customer Profile, Active Claim, historical Claim History records, and the Pinecone Policy Exclusions all at once and bundles them into a single context object.

2. **Deterministic Pre-Computation**
   To prevent hallucination on rigid mathematical logic, the script pre-computes specific Boolean signals (e.g., "Has the customer breached the frequent claimant threshold?"). These signals are injected directly into the prompt.

3. **Monolithic Prompting (`nma_src/agents/nma_agent.py`)**
   The massive context payload is injected into a single, highly complex system prompt. The LLM is forced via a strict Pydantic schema to simultaneously assign a fraud risk score, determine policy coverage, apply exclusions, and calculate the final financial payout in one shot.

4. **Deterministic Overrides**
   Post-LLM execution, the Python script applies hard deterministic overrides. If mathematical conditions are breached (e.g. remaining limit is 0, or hard fraud signals fire), the script overrides the LLM and forces the payout to $0.00.

## Tech Stack
* **LLM Provider**: Ollama (Running locally via `llama3:latest`)
* **Context Retrieval**: PyMongo (MongoDB) & Pinecone (Vector database)
* **Structured Validation**: Pydantic v2
* **Evaluation Framework**: Custom Python harness mirroring the MAS `evals` module.

## Usage

*Important: Run these commands from the root `insurance-agent` directory.*

### Manual Claim Testing
To test the single agent on a specific Claim ID from your MongoDB interactively:
```bash
python insurance-agent-NMA/nma_src/main.py
```

### Run the Evaluation Benchmark
To run the Single-Agent System against the 50-claim dataset and calculate accuracy, latency, and hallucination metrics:
```bash
python insurance-agent-NMA/evals/run_evals_nma.py
```
