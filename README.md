# Insurance Claim Adjudication: Multi-Agent vs Single-Agent

## Purpose
This repository is a production-grade evaluation framework designed to benchmark a **Multi-Agent System (MAS)** against a traditional **Single-Agent System (NMA)** for end-to-end car insurance claim processing. 

Insurance adjudication requires cross-referencing user narratives with historical data and complex policy exclusions. This project empirically measures whether breaking cognitive tasks into a graph of specialized, parallel agents (MAS) yields statistically significant improvements in accuracy, hallucination rates, and fraud detection over a monolithic LLM prompt (NMA).

## Tech Stack
* **LLM Provider**: Ollama (Running locally via `llama3:latest`)
* **Agent Framework**: LangGraph (StateGraph for sequential/parallel routing)
* **Databases**: 
  * **MongoDB Atlas**: Document store for customer profiles, claim history, and active claims.
  * **Pinecone**: Vector database for RAG (semantic policy matching).
* **Validation**: Pydantic v2 (Strict JSON schema enforcement)
* **Embeddings**: HuggingFace `sentence-transformers/all-MiniLM-L6-v2`

## How They Were Made & Structure

### The Multi-Agent System (MAS)
Located in `src/agents/`. This architecture splits the cognitive load into discrete, specialized nodes coordinated by a LangGraph StateMachine:
- **Verification Agent**: Validates initial claim structure and OCR data.
- **Parallel Analysis**:
  - **Policy Agent**: Fetches the vector policy and calculates coverage limitations.
  - **Fraud Agent**: Checks historical claim data and cross-references active claims to detect anomalies like collusion or staging.
- **Decision Agent**: Synthesizes the signals to approve/deny and calculate the final payout.

### The Single-Agent System (NMA)
Located in `insurance-agent-NMA/`. Built as a baseline.
- It uses a massive context fetcher (`context_fetcher.py`) to aggregate all Pinecone and MongoDB data upfront.
- A single, monolithic prompt forces the LLM to simultaneously detect fraud, determine coverage, and calculate payouts in one shot.

## Folder Structure
```
d:\Projects\insurance-agent\
├── src/                  # MAS Logic (LangGraph orchestration and Agent definitions)
├── insurance-agent-NMA/  # Baseline NMA Logic and isolated NMA evaluation runner
├── evals/                # Shared metrics, ground-truth loading, and MAS evaluation runner
├── scripts/              # Synthetic data generation and seeding scripts
├── .env                  # Configuration keys (MongoDB, Pinecone)
└── README.md             # This main documentation file
```

## How the Database Was Created & Where Stuff is Stored
To test agents realistically, they must operate against a live, stateful environment rather than isolated text files. 
- **Database Creation (`scripts/seed_data.py`)**: A Python script using `Faker` generates 50 fully interconnected, synthetic records across Customer Profiles, Claim History, and Active Claims. It deliberately injects 5 specific fraud scenarios (e.g., frequent claimants, collusion rings, semantic exclusions) to test the agents' reasoning limits.
- **Storage Locations**:
  - **MongoDB Atlas**: Stores the structured, transactional data (Customers, History, Active Claims).
  - **Pinecone**: Stores the vectorized Insurance Policies.

## How Chunking Was Done
Due to the relatively short length of car insurance policies, the `seed_data.py` script embeds the entire policy string as a **single Pinecone document** (no text chunking is performed at ingestion). 
However, during MAS execution, the `policy_agent.py` performs **dynamic in-memory chunking** on the retrieved policy: it splits the `exclusions_text` by newlines into individual clauses and batch-encodes them to prevent cosine-similarity dilution, ensuring highly accurate semantic matching against the claim narrative.

## How to Run

### 1. Environment Setup
Install dependencies and configure your `.env`:
```env
PINECONE_API_KEY=your_pinecone_key
MONGODB_URL=mongodb://localhost:27017 # or Atlas URL
```
*Note: Ensure Ollama is running locally with the `llama3:latest` model downloaded (`ollama pull llama3`).*

### 2. Seed the Database
Run this once to wipe the database and generate the 50 ground-truth claims:
```bash
python scripts/seed_data.py
```

### 3. Run the Actual MAS Application
To run the Multi-Agent System pipeline directly against an active claim ID from your MongoDB:
```bash
python src/main.py
```

### 4. Run the Evaluations
The evaluation runners execute the agents against the 50-claim dataset and output detailed JSON reports to the `evals/results/` folder containing accuracy, precision/recall, latency, and hallucination metrics.

**Evaluate MAS:**
```bash
python -m evals.run_evals
```

**Evaluate NMA:**
```bash
python insurance-agent-NMA/evals/run_evals_nma.py
```
