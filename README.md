# Insurance Claim Adjudication: Multi-Agent System vs Single-Agent Baseline

## Overview

This repository is a production-grade research and evaluation framework that benchmarks a **Multi-Agent System (MAS)** against a **Single-Agent Baseline (NMA)** for end-to-end car insurance claim adjudication. The core research question is whether decomposing the cognitive load of claim processing into a graph of specialized, parallel agents yields statistically significant improvements in accuracy, fraud detection, and semantic reasoning over a monolithic large language model prompt.

Insurance adjudication is a particularly demanding domain for this comparison. A single claim requires the system to simultaneously cross-reference structured transactional data, apply unstructured policy language to a free-text incident narrative, detect behavioral anomalies across a customer's claim history, and produce a financially correct payout calculation. These are cognitively distinct tasks that a single LLM prompt must handle all at once, whereas a multi-agent architecture can assign each task to a specialized node with focused context and a narrowly scoped prompt.

---


https://github.com/user-attachments/assets/d77969fd-dca6-46eb-b85a-99ad8e31554e




## Tech Stack

| Component | Technology |
|---|---|
| LLM Provider | Ollama (local inference, `llama3:latest`) |
| Agent Orchestration | LangGraph `StateGraph` |
| Document Store | MongoDB Atlas |
| Vector Database | Pinecone (serverless, cosine similarity) |
| Embeddings | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` |
| Schema Validation | Pydantic v2 |
| Synthetic Data Generation | Faker |
| Evaluation & Metrics | Custom Python harness |
| API Layer | FastAPI |

---

## Why Ollama and Local Inference

All LLM calls in both systems use `llama3:latest` running locally via Ollama. This was a deliberate choice for three reasons. First, it eliminates external rate limits and API costs during the evaluation sweep across 50 claims with multiple consistency runs. Second, it ensures both the MAS and NMA are evaluated under identical model conditions with no variance introduced by cloud provider load or model versioning. Third, it allows the evaluation harness to run concurrent batches without hitting rate-limit backoff logic that would inflate latency measurements.

The tradeoff is that `llama3` at 4-bit quantization is a weaker reasoner than frontier models, which makes the benchmark harder and more meaningful — if the MAS architecture produces gains over the NMA even with a constrained local model, those gains are likely to amplify further with a stronger model.

---

## Why Real Data Was Not Used

The evaluation dataset is entirely synthetic. Real insurance claim data was not used for the following reasons.

**Privacy and regulatory compliance.** Insurance claim records contain personally identifiable information, medical details, and financially sensitive data. Using real data would require anonymization pipelines, data sharing agreements, and compliance review that are outside the scope of this research project.

**Ground truth labelling.** A rigorous evaluation requires a dataset where the correct adjudication outcome is known with certainty for every claim. Real historical claims have outcomes that depend on adjuster judgment, legal precedent, and policy interpretation that varies by insurer. There is no clean ground truth label. Synthetic data allows the seeding script to embed the correct label (`fraud_scenario`) at generation time, making evaluation deterministic and reproducible.

**Controlled fraud scenario injection.** To measure fraud detection capability, the dataset must contain a known number of fraud cases of specific types. Real datasets have unknown or disputed fraud rates and uneven coverage of scenario types. The synthetic generator deliberately creates equal representation of all five fraud patterns so that per-scenario accuracy can be measured cleanly.

**Reproducibility.** Any researcher or reviewer can re-run `scripts/seed_data.py` to regenerate the exact same logical dataset structure (with different random UUIDs and names from Faker, but identical scenario distribution and signal patterns). This is not possible with real proprietary data.

---

## Repository Structure

```
insurance-agent/
├── src/                            # Multi-Agent System (MAS)
│   ├── agents/
│   │   ├── verification_agent.py   # Document validation and OCR parsing
│   │   ├── policy_agent.py         # Pinecone RAG + semantic exclusion matching
│   │   ├── fraud_agent.py          # Five-signal fraud detection engine
│   │   └── decision_agent.py       # Final adjudication and payout calculation
│   ├── graph/
│   │   ├── graph.py                # LangGraph StateGraph pipeline definition
│   │   └── state.py                # ClaimState TypedDict
│   ├── api/
│   │   └── routes.py               # FastAPI REST endpoint
│   ├── tools/
│   │   ├── llm_client.py           # Ollama client factory (sync + async)
│   │   └── mongo_client.py         # MongoDB and Pinecone connection singletons
│   └── main.py                     # MAS entry point and pre-warming
│
├── insurance-agent-NMA/            # Single-Agent Baseline (NMA)
│   ├── nma_src/
│   │   ├── agents/
│   │   │   └── nma_agent.py        # Monolithic LLM adjudicator
│   │   ├── tools/
│   │   │   └── context_fetcher.py  # Bulk context aggregation from MongoDB + Pinecone
│   │   ├── schemas/
│   │   │   └── nma_schema.py       # Pydantic output schema
│   │   └── main.py                 # NMA entry point
│   └── evals/
│       └── run_evals_nma.py        # NMA evaluation runner
│
├── evals/                          # Shared evaluation infrastructure
│   ├── run_evals.py                # MAS evaluation orchestrator
│   ├── ground_truth.py             # Dynamic label loading from MongoDB
│   ├── metrics.py                  # Pure-Python metric calculations
│   ├── llm_judge.py                # Groundedness and hallucination scoring
│   ├── chaos.py                    # OCR corruption robustness testing
│   ├── reporter.py                 # JSON report and Matplotlib dashboard
│   └── results/                    # Generated output directory (git-ignored)
│
├── scripts/
│   └── seed_data.py                # Synthetic dataset generation and DB seeding
│
├── evals/demo.py                   # Interactive demo tool (preset + manual modes)
├── requirements.txt
└── README.md
```

---

## Database Schema

The evaluation environment uses four collections and indexes across two databases. Every record is linked by two UUIDs generated per customer: `customer_id` and `policy_id`. This guarantees referential integrity across all collections with no orphaned records.

### MongoDB Collections

**Customer_Profiles**

| Field | Type | Description |
|---|---|---|
| `customer_id` | UUID string | Primary key |
| `full_name` | string | Faker-generated name |
| `tenure_months` | int | Policy tenure (1 to 240 months) |
| `ncd_tier` | float | No-Claims Discount tier (0.0 to 0.5) |
| `risk_rating` | string | "Safe", "Watchlist", or "High Risk" |
| `policy_id` | UUID string | Foreign key to Active_Claims and Pinecone |

**Claim_History**

| Field | Type | Description |
|---|---|---|
| `history_id` | UUID string | Primary key |
| `customer_id` | UUID string | Foreign key to Customer_Profiles |
| `incident_type` | string | One of: Rear-end, Theft, Vandalism, Hit and Run, Total Loss |
| `payout_amount` | float | Approved payout (0.0 if Denied or Fraud_Flagged) |
| `claim_status` | string | "Approved", "Denied", or "Fraud_Flagged" |
| `incident_date` | date string | ISO date, up to 2 years in the past |

**Active_Claims**

| Field | Type | Description |
|---|---|---|
| `claim_id` | UUID string | Primary key |
| `policy_id` | UUID string | Foreign key to Customer_Profiles and Pinecone |
| `customer_id` | UUID string | Foreign key to Customer_Profiles |
| `incident_type` | string | Category of incident |
| `incident_date` | date string | Date of the loss event |
| `narrative` | string | Free-text incident description |
| `ocr_extraction` | object | Structured fields extracted from the repair document (see below) |
| `estimated_loss` | float | Claimant's declared loss amount |
| `fraud_scenario` | string | Ground-truth label (hidden from agents during processing) |

The `ocr_extraction` sub-document contains: `PolicyNumber`, `ClaimantName`, `LossDate`, `RepairShopName`, and `TotalEstimate`. The `TotalEstimate` field is the figure extracted from the repair shop's document, which for staged accident scenarios is deliberately set to $200 while `estimated_loss` is $8,000–$20,000.

### Pinecone Index

One vector per customer representing their entire insurance policy. Each vector stores the 384-dimensional embedding of the full policy text and carries the following metadata: `policy_id`, `customer_id`, `customer_name`, `coverage_scope`, `policy_limit`, `aggregate_limit`, `deductible`, `exclusions`, and `total_historical_payout`.

---

## Synthetic Dataset Design

### Scale and Distribution

50 customers are generated, each with one Active_Claim. The fraud scenario distribution is:

| Scenario | Count | Expected Decision |
|---|---|---|
| `normal` | 18 | Approved |
| `collusion_ring` | 8 | Denied |
| `semantic_exclusion` | 10 | Denied |
| `staged_accident` | 5 | Denied |
| `frequent_claimant` | 5 | Denied |
| `aggregate_breach` | 4 | Denied |

### Fraud Scenario Construction

**Frequent Claimant.** Customers in this group have claim histories generated with a bias toward `Denied` and `Fraud_Flagged` statuses. The seeder calls `gen_history()` with `fraud_heavy=True`, ensuring at least 3 denied or flagged historical records. The active claim is a vandalism incident with a plausible narrative and loss amount. The fraud signal is detected purely by cross-referencing claim history, not by anything in the narrative itself.

**Collusion Ring.** Eight unrelated customers, with no connection in their profiles or histories, all have the same repair shop (`Apex AutoBody & Collision`) injected into their `ocr_extraction.RepairShopName` field. Each narrative is independently plausible. The signal exists only at the shop-name level and requires the agent to maintain awareness of a flagged shop list.

**Staged Accident.** The narrative describes an extreme multi-vehicle disaster (14 vehicles, emergency services, ambulances, six-hour road closure) while the `ocr_extraction.TotalEstimate` is set to $200. The `estimated_loss` is $8,000–$20,000. Detection requires comparing the narrative's implied severity against the repair document figure — a reasoning step that cannot be done with keyword matching alone.

**Semantic Exclusion.** This is the most technically demanding scenario. Ten narratives describe incidents that fall under policy exclusions (track racing, off-road driving, DUI, rideshare/commercial use) but use natural language that does not contain the exact exclusion keyword. For example, a narrative about "a closed circuit track event" must be matched to a policy exclusion that says "Street Racing." The seeder uses a mapping table (`SEMANTIC_EXCLUSION_TEMPLATE_MAP`) to ensure each narrative is assigned a policy whose exclusions text is guaranteed to contain the semantically appropriate clause.

**Aggregate Breach.** The `estimated_loss` for these claims is calculated as `aggregate_limit - total_historical_payout + random offset`, ensuring the claim amount exceeds the customer's remaining lifetime policy limit. Detection is a pure arithmetic check, but the system must retrieve the correct policy limits and historical payout totals to perform it.

**Normal Claims.** Legitimate rear-end collisions, thefts, and weather damage with narratives that are plausible but contain no fraud signals. Loss amounts are capped at `min(policy_limit * 0.9, $12,000)` to prevent normal claims from accidentally triggering high-value thresholds. These test the system's specificity — the ability to approve a valid claim without inventing fraud.

### Why No Text Chunking

Insurance policies in this dataset are short enough (one structured paragraph per customer) to be stored as single Pinecone documents without chunking. The `exclusions` field within the policy is a single string of comma-separated clauses. At retrieval time, the MAS policy agent performs in-memory splitting of this exclusions string into individual clauses and batch-encodes them separately to prevent cosine-similarity dilution. This is described in detail in the Architecture section below.

---

## Multi-Agent System (MAS) Architecture

The MAS is implemented as a LangGraph `StateGraph` that routes each claim through a fixed pipeline. All state is passed through a typed `ClaimState` dictionary.

```
document_verification
        |
   (conditional)
   /            \
failure_node   parallel_analysis
                (policy_agent || fraud_agent)
                        |
                    decision
                        |
                       END
```

### Verification Agent

Fetches the claim from `Active_Claims` and the customer profile from `Customer_Profiles`. Parses the `ocr_extraction` sub-document into a structured `OcrExtraction` Pydantic model. Cross-validates the OCR fields against the profile (name match, policy number match). Makes zero LLM calls — this is entirely deterministic.

Discrepancies such as name or policy number mismatches are recorded as **warnings** rather than errors. This is a deliberate design decision: OCR quality issues are real in production environments and should not cause a claim to be hard-denied before the policy and fraud agents have evaluated it. Only truly fatal conditions (claim not found in the database, or no customer profile exists) are written to `state["errors"]`, which triggers routing to `failure_node`.

### Parallel Analysis Node

The policy agent and fraud agent run concurrently via `asyncio.gather()`. Both agents depend only on `sanitized_data` from the verification step, so they are fully independent. Errors from both agents are merged into a true union — verification errors are not overwritten by agent-specific errors.

### Policy Agent

Embeds the claim narrative using `all-MiniLM-L6-v2` and queries Pinecone with a filter on `policy_id` to retrieve the correct policy. If the exact-match filter returns no results, it falls back to an unfiltered top-1 query with a policy ID check.

The key innovation in this agent is **per-clause exclusion matching**. Rather than computing cosine similarity between the narrative vector and the entire `exclusions` string (which is diluted across many unrelated clauses), the agent splits `exclusions_text` on newlines into individual clauses, batch-encodes all clauses, and takes the maximum cosine similarity across the clause matrix. This produces a more accurate similarity signal for the specific exclusion that applies to the narrative.

The exclusion decision gate combines two signals: the LLM's quoted exclusion reason, and a substring verification check that confirms the quoted text actually appears in the stored exclusions string. This prevents hallucinated exclusions from firing while still allowing the LLM to identify exclusions that the similarity metric would miss.

Deterministic calculations (remaining limit, aggregate breach detection) are performed in Python, not delegated to the LLM. The LLM is only asked to produce a coverage reasoning string and quote the applicable exclusion clause verbatim.

### Fraud Agent

Pre-computes five deterministic signals before making any LLM call:

1. **Frequent claims flag** — counts `Denied` and `Fraud_Flagged` records in `Claim_History`. Fires if count >= 3.
2. **Collusion flag** — checks if `COLLUSION_SHOP` ("Apex AutoBody & Collision") is a case-insensitive substring of the `RepairShopName` field.
3. **Claim velocity** — counts claims in the last 365 days.
4. **High-value high-risk** — fires if customer is "High Risk" and `estimated_loss` > $20,000.
5. **Staging pre-signal** — fires if `ocr_total_estimate` < $1,000.

These signals are injected into the LLM prompt as pre-computed facts. The LLM is asked to rate the narrative's physical severity on a scale of 1–10, then synthesize all signals into a final risk level. The staging flag is confirmed post-LLM only if both the narrative severity score is >= 7 and the staging pre-signal fired — this prevents the LLM from triggering staging on low-severity incidents with small OCR estimates.

Post-LLM, the agent applies escalation and clamping rules: spurious LLM HIGH escalations with no deterministic backing are clamped to MEDIUM, and confirmed staging or collusion flags override any lower LLM assessment.

### Decision Agent

Receives the policy verdict and fraud report and makes the final approve/deny decision. The LLM is given explicit financial figures (`estimated_loss`, `deductible`, `remaining_limit`) and asked to produce exactly three numbered reasoning sentences: (1) policy coverage finding, (2) fraud assessment, (3) financial outcome with arithmetic. This structured format was designed to improve hallucination judge scoring by giving the LLM three independently verifiable atomic claims to produce.

---

## Single-Agent Baseline (NMA) Architecture

The NMA processes each claim in a single LLM call preceded by a bulk data fetch. It was built to test whether a carefully engineered monolithic prompt can achieve comparable results to the MAS without agent decomposition.

### Context Fetcher

The `context_fetcher.py` module performs one round of database queries upfront: the active claim from `Active_Claims`, the customer profile from `Customer_Profiles`, the full claim history from `Claim_History`, and the policy from Pinecone. All of this is assembled into a single context dictionary. The Pinecone query uses the full narrative as the embedding query with a `policy_id` filter, retrieving the entire policy blob as one document.

### Deterministic Pre-Computation

To prevent the LLM from hallucinating on mathematical logic, the NMA pre-computes the same five fraud signals as the MAS fraud agent and injects them as explicit `YES/NO` flags into the prompt. It also pre-computes `remaining_limit` and formats the full claim history as a plain-text summary.

### Monolithic Prompt

The single LLM call receives the full context and must simultaneously assign a fraud risk score (1–10), detect which signals apply, determine policy coverage, check for exclusion matches, and calculate the payout. Post-LLM, the Python code applies a `must_deny` override that forces a denial if the collusion or staging signal fired, regardless of the LLM's output.

### Key Architectural Differences vs MAS

| Dimension | MAS | NMA |
|---|---|---|
| LLM calls per claim | 3 (policy + fraud + decision) | 1 |
| Exclusion matching | Per-clause vector similarity + substring verification | Single narrative-vs-policy-blob similarity |
| Fraud reasoning | Dedicated agent with history analysis | Prompt injection of pre-computed signals |
| Staging detection | LLM severity score + deterministic gate | Prompt injection only (no LLM severity score) |
| Error isolation | Agents fail independently | Single failure aborts entire adjudication |
| Post-LLM overrides | Risk escalation + clamping rules | must_deny for collusion and staging only |

---

## Evaluation Framework

Both systems are evaluated against the same 50-claim ground-truth dataset using a shared evaluation harness located in `evals/`. Ground-truth labels are loaded dynamically from MongoDB by reading the `fraud_scenario` field planted by the seeder — no labels are hardcoded in the evaluation code.

### Metrics

**Decision Accuracy** compares `final_decision.approved` against the expected decision derived from `fraud_scenario`. All non-normal scenarios expect `Denied`.

**Fraud Precision / Recall / F1** treats `fraud_report.risk_score == "High"` as the predicted positive and `is_fraud` (any non-normal, non-aggregate-breach scenario) as the actual positive.

**Straight-Through Processing (STP) Rate** measures the percentage of claims that reached the decision node without being short-circuited by the failure node. A claim hits the failure node only on fatal verification errors.

**Groundedness Score** uses an LLM-as-judge (also running on Ollama) to rate whether the `policy_verdict.coverage_reasoning` is supported by the retrieved policy text. Scored 1 (unsupported) to 5 (fully grounded) on a sample of 5 claims per run.

**Hallucination Rate** scores each sentence in `final_decision.step_by_step_reasoning` as `supported`, `unsupported`, or `uncertain`. The hallucination rate is `unsupported_count / total_sentences`.

**Consistency** runs each claim K times independently and measures the rate of identical decisions across runs. Used to detect non-determinism in the LLM.

**Chaos / Robustness** patches the MongoDB `ocr_extraction` field directly before running the pipeline (and restores it after in a `try/finally` block), simulating OCR read failures at 10%, 20%, and 30% field corruption rates. This tests whether the system degrades gracefully when input data quality decreases.

### Evaluation Runners

The MAS evaluation runner (`evals/run_evals.py`) processes claims in concurrent batches using `asyncio.gather()` with a configurable concurrency level (default 5). The NMA runner (`insurance-agent-NMA/evals/run_evals_nma.py`) mirrors this structure with concurrency defaulting to 8, since the NMA fires only one LLM call per claim versus the MAS's three.

Both runners output a JSON report and a 6-panel Matplotlib dashboard to `evals/results/`.

---

## How to Run

### 1. Prerequisites

Install dependencies and configure your `.env` file in the repo root:

```env
PINECONE_API_KEY=your_pinecone_api_key
MONGODB_URL=mongodb+srv://<user>:<password>@<cluster>.mongodb.net/
```

Ensure Ollama is running locally with the llama3 model downloaded:

```bash
ollama pull llama3
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

### 2. Seed the Database

Run once to drop existing collections and regenerate the 50-claim ground-truth dataset:

```bash
python scripts/seed_data.py
```

### 3. Run the MAS Pipeline on a Single Claim

```bash
python src/main.py <claim_id>
```

### 4. Run the Interactive Demo

```bash
# Preset scenarios (recommended for demos)
python evals/demo.py --preset normal
python evals/demo.py --preset staged_accident
python evals/demo.py --preset collusion_ring
python evals/demo.py --preset semantic_exclusion

# Manual input
python evals/demo.py --manual

# List all seeded claim IDs grouped by scenario
python evals/demo.py --list
```

### 5. Run Evaluations

Evaluate the MAS:

```bash
python -m evals.run_evals
```

Evaluate the NMA:

```bash
python insurance-agent-NMA/evals/run_evals_nma.py
```

Fast evaluation (no LLM judge, no chaos, single pass):

```bash
python -m evals.run_evals --k 1 --no-llm-judge
```

### 6. Start the API Server

```bash
uvicorn src.api.routes:app --reload
```

---

## Known Limitations

**Semantic exclusion accuracy is the hardest problem.** Both systems struggle with this scenario relative to others. The fundamental challenge is that natural language in an incident narrative and formal language in a policy exclusion clause occupy different regions of the embedding space. The MAS mitigates this with per-clause matching and the LLM as a semantic bridge, but llama3's reasoning on this task is not reliable for all narrative phrasings.

**llama3 is non-deterministic under concurrent load.** Even with `temperature=0`, Ollama under concurrent requests can produce slightly different outputs for the same prompt. This is why the evaluation framework includes a consistency metric.

**The LLM judge uses the same model as the pipeline.** Groundedness and hallucination scoring is performed by a second llama3 instance running on Ollama. This creates a circularity where the judge and the subject are the same model. A stronger external judge (e.g. GPT-4) would produce more reliable quality scores.

**Latency is high.** Mean latency of 194 seconds per claim for the MAS on local hardware reflects the sequential LangGraph overhead combined with local Ollama inference time. The concurrent batch runner reduces wall-clock evaluation time significantly but individual claim latency is unchanged.
