# Multi-Agent vs. Single-Agent Insurance Claim Adjudication System

## Overview
This repository contains a production-grade, stateful evaluation framework designed to benchmark a **Multi-Agent System (MAS)** against a **Single-Agent System (NMA)** for end-to-end car insurance claim processing. 

Insurance claim adjudication is a complex domain that requires cross-referencing user narratives with physical evidence, historical claims data, and dense policy exclusions. The primary goal of this project is to empirically measure whether breaking this complex task into a graph of specialized, parallel agents (MAS) yields statistically significant improvements in accuracy, Straight-Through Processing (STP) rates, and fraud detection over a traditional monolithic "all-in-one" LLM prompt (NMA).

---

## The Goal
1. **Automate Adjudication**: Automatically parse a car insurance claim and determine if it should be approved or denied based on fraud signals, aggregate limits, and semantic policy exclusions.
2. **Prevent Hallucination**: Use deterministic Python fallbacks and pre-computations so the LLM cannot approve a claim that violates hard mathematical rules (e.g., claiming more than the remaining limit).
3. **Benchmarking**: Provide a 50-claim "Ground Truth" testbed to evaluate the latency, token efficiency, and precision/recall of the MAS vs the NMA.

---

## Tech Stack
* **Agent Orchestration Framework**: LangGraph (StateGraph for sequential/parallel routing and cyclic flows in MAS)
* **LLM Provider**: Groq (`llama-3.3-70b-versatile` for deep reasoning tasks at high speed)
* **Vector Database (RAG)**: Pinecone (Used for semantic matching between the claim narrative and dense policy exclusion clauses)
* **Document/Relational Database**: MongoDB (Serves as the internal "Ground Truth" for claim history, customer profiles, and active claims)
* **Validation**: Pydantic v2 (Enforces strict JSON schemas for all LLM outputs)
* **Embedding Model**: HuggingFace `sentence-transformers/all-MiniLM-L6-v2` (for RAG vectorization)

---

## Implementation Strategy

### 1. Synthetic Data & "Ground Truth"
A major limitation of standard AI demos is the reliance on isolated prompt inputs. To make this production-grade, we integrated a real MongoDB database layer seeded with thousands of relational records (Policyholders, Vehicles, Past Claims, Fraud Signals). The system evaluates claims against this persistent state.

### 2. The Multi-Agent System (MAS) Paradigm
Found in `src/agents`. This architecture splits the cognitive load into discrete, specialized nodes:
- **Verification Agent**: Extracts and validates the raw claim and OCR data.
- **Parallel Analysis**:
  - **Policy Agent**: Performs Pinecone RAG lookups to check coverage scope and semantic exclusions.
  - **Fraud Agent**: Queries MongoDB for the customer's claim history to detect frequent claimants, collusion rings, and staging anomalies.
- **Decision Agent**: Synthesizes the outputs of the parallel nodes to calculate the final payout and approve/deny the claim.

### 3. The Single-Agent System (NMA) Paradigm
Found in `insurance-agent-NMA/nma_src`. This architecture serves as the baseline comparison. It features a single `context_fetcher.py` script that aggregates all MongoDB and Pinecone data into one massive payload. A single prompt forces the LLM to simultaneously detect fraud, determine policy coverage, and calculate the final financial payout in one shot.

---

## Setup Instructions

### 1. Prerequisites
- Python 3.11+
- A local **MongoDB** instance running on `localhost:27017`
- A `.env` file in the root directory:
  ```env
  GROQ_API_KEY=your_groq_key
  PINECONE_API_KEY=your_pinecone_key
  MONGODB_URL=mongodb://localhost:27017
  ```

### 2. Environment Setup
Install the dependencies:
```bash
pip install -r requirements.txt
```

### 3. Database Initialization
Seed the local MongoDB instance with the 50 synthetic claims, customer profiles, and historical records:
```bash
python scripts/seed_data.py
```

### 4. Running the Evaluations
The evaluation runners execute the agents against the 50-claim dataset and output detailed JSON reports containing accuracy, precision/recall, latency, and token metrics.

To evaluate the Multi-Agent System (MAS):
```bash
python -m evals.run_evals
```

To evaluate the Single-Agent System (NMA):
```bash
python insurance-agent-NMA/evals/run_evals_nma.py
```

Results and metric reports will be populated in the `evals/results` folder.
