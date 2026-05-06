# Car Insurance MAS — Test Data Seeding Guide

A practical guide for new engineers joining the Car Insurance Claim Processing project. This document explains what the seeding script does, why each decision was made, and how to run it from scratch.

---

## What Is This?

This project is a **Role-Specialized Multi-Agent System (MAS)** for processing car insurance claims. Multiple AI agents work together — each with a specific job — to assess incoming claims, detect fraud, check policy coverage, and make payout decisions.

Before you can test those agents, you need realistic data in the databases. That is what `seed_data.py` does: it generates **50 interconnected, synthetic records** across two databases and populates them in one shot.

---

## The Two Databases (and Why Two?)

### MongoDB Atlas — Transactional / Structured Data
MongoDB stores everything that looks like a traditional database table: customer profiles, claim history, and the active claims the agents will process. It is good at structured queries like *"find all claims for customer X"* or *"how many fraud flags does this person have?"*

### Pinecone — Vector / Semantic Search
Pinecone stores insurance policy documents as **vector embeddings** — mathematical representations of text that capture meaning, not just keywords. This powers the RAG (Retrieval-Augmented Generation) step where an agent fetches the relevant policy for a claim and checks whether the incident is actually covered.

For example, a narrative that says *"driving at high speed on a closed circuit"* does not contain the word "racing" — but its vector embedding is semantically close to the exclusion clause *"Street Racing"*, so Pinecone will surface the match. A keyword search would miss this entirely.

---

## Data Schema

Four collections/indexes are created:

```
MongoDB
├── Customer_Profiles      — Who the customer is and their risk level
├── Claim_History          — Their past claims and outcomes
└── Active_Claims          — The 50 new claims agents will process today

Pinecone
└── insurance-policies     — One policy document per customer, vectorized
```

### How the IDs Connect

Every record is linked by two IDs generated at the start of each customer's creation:

```
customer_id (UUID)  ──→  Customer_Profiles
                    ──→  Claim_History (many records)
                    ──→  Active_Claims
                    ──→  Pinecone metadata

policy_id (UUID)    ──→  Insurance_Policies (Pinecone vector ID)
                    ──→  Active_Claims
                    ──→  Customer_Profiles
```

Both IDs are created once per customer and reused everywhere. This guarantees 100% referential integrity — no orphaned records, no broken links.

---

## How the Data Is Generated

### Step 1 — Customer Profiles

50 customers are created using the `faker` library for realistic names. Each customer gets:

- A `tenure_months` (how long they have been insured, 1–240 months)
- An `ncd_tier` (No-Claim Discount, a float from 0.0 to 0.5 — higher means they have been claim-free longer)
- A `risk_rating` of Safe, Watchlist, or High Risk — derived from how many denied/fraud-flagged entries exist in their history

### Step 2 — Claim History

Each customer gets a randomly sized batch of past claims. These are used by agents to spot patterns — someone with four denied claims in a row is a very different risk profile than a first-time claimant.

For fraud-scenario customers, the history is deliberately loaded with `Denied` and `Fraud_Flagged` statuses.

### Step 3 — Insurance Policies (Pinecone)

A policy document is assembled as a plain English string:

```
"Policy ID: abc123. Customer: Jane Doe. Coverage: Comprehensive coverage 
including glass breakage... Policy Limit: $25,000. Aggregate Limit: $75,000. 
Deductible: $500. Exclusions: Street Racing, DUI, Off-roading..."
```

This string is passed through `sentence-transformers` (model: `all-MiniLM-L6-v2`) to produce a **384-dimension vector**. That vector, plus all the key fields as metadata, is upserted into Pinecone using the `policy_id` as the record ID.

When an agent later queries Pinecone with a claim narrative, it gets back the most semantically similar policy — including all the structured metadata it needs to make a coverage decision.

### Step 4 — Active Claims

One active claim is created per customer. This is the record that enters the MAS pipeline. It contains:

- `narrative` — A paragraph describing what happened, written to challenge NLP reasoning
- `ocr_extraction` — Simulates what an OCR tool would extract from a submitted repair document (policy number, claimant name, loss date, shop name, repair estimate)
- `estimated_loss` — The amount the claimant is requesting
- `fraud_scenario` — A label for your reference during testing (agents do not see this field)

---

## Fraud & Edge Case Injection

40% of records (the first 22 customers) are deliberately crafted to test specific agent failure modes. The remaining 28 are clean, normal claims used as a baseline.

### 1. The Frequent False Claimant (Records 0–4)
**What it is:** Customers who already have 3–5 `Denied` or `Fraud_Flagged` entries in their history, now submitting a brand new Vandalism claim.

**What it tests:** Whether the agent checks claim history before approving, and whether it escalates repeat offenders rather than processing the claim normally.

**Signal in data:** High count of non-Approved statuses in `Claim_History`. New claim `incident_type` is always `Vandalism`.

---

### 2. The Collusion Ring (Records 5–9)
**What it is:** Five completely unrelated customers who all list the same repair shop — `Apex AutoBody & Collision` — in their `ocr_extraction.RepairShopName`.

**What it tests:** Cross-claim fraud detection. No single claim looks suspicious on its own. The fraud only appears when an agent compares repair shop names across multiple active claims simultaneously.

**Signal in data:** `ocr_extraction.RepairShopName == "Apex AutoBody & Collision"` on five records belonging to five different customers with no other connection.

---

### 3. The Staged Accident (Records 10–14)
**What it is:** The narrative describes a catastrophic multi-vehicle pileup — dozens of cars, emergency services on scene for hours. But the OCR-extracted repair estimate is just **$200**.

**What it tests:** Whether the agent can detect the mismatch between a described incident's severity and its claimed financial loss. A genuine 12-car pileup does not result in a $200 repair bill.

**Signal in data:** `narrative` contains dramatic, high-severity language. `ocr_extraction.TotalEstimate == 200.0` while `estimated_loss` is set normally high.

---

### 4. The Semantic Exclusion (Records 15–17)
**What it is:** The narrative describes driving on a closed circuit or participating in a timed lap event — essentially street racing — without ever using the word "racing."

**What it tests:** Whether the RAG step correctly retrieves the policy and whether the agent can semantically match the narrative to the exclusion clause, not just keyword-match it.

**Signal in data:** Narrative uses language like *"closed circuit track"* or *"timed lap competition."* The policy exclusions contain *"Street Racing."* These are semantically similar in vector space but lexically different.

---

### 5. The Aggregate Limit Breach (Records 18–21)
**What it is:** The `estimated_loss` on the new claim, when added to the total `payout_amount` across historical approved claims, exceeds the policy's `aggregate_limit` (the lifetime maximum payout cap).

**What it tests:** Whether the agent performs the full financial calculation — not just checking if the single claim is under the `policy_limit`, but whether the customer has already consumed most of their lifetime coverage.

**Signal in data:** `estimated_loss` is deliberately computed as: `aggregate_limit - total_historical_payout + a random overage`. The math is always a breach.

---

## Running the Script

### Prerequisites

```bash
pip install pymongo pinecone-client faker sentence-transformers python-dotenv
```

### Environment Variables

Create a `.env` file in the same directory as the script:

```
MONGODB_URL=mongodb+srv://<user>:<password>@<cluster>.mongodb.net/
PINECONE_API_KEY=your-pinecone-api-key
```

### Run

```bash
python seed_mas_data.py
```

### Expected Output

```
[MongoDB] Inserted 50 Customer_Profiles
[MongoDB] Inserted ~120 Claim_History records
[MongoDB] Inserted 50 Active_Claims
[Pinecone] Upserted 50 Insurance_Policies vectors

[Summary] Fraud scenario distribution:
  frequent_claimant: 5 records
  collusion_ring: 5 records
  staged_accident: 5 records
  semantic_exclusion: 3 records
  aggregate_breach: 4 records
  normal: 28 records
```

> **Note:** The script drops and recreates the three MongoDB collections every time it runs. This is intentional — it keeps your test environment clean and repeatable. Pinecone uses `upsert`, so re-running will overwrite existing vectors without creating duplicates.

---

## What To Do After Seeding

Once the data is loaded, your agents can be tested against `Active_Claims`. A typical agent pipeline looks like:

1. **Intake Agent** — Reads a claim from `Active_Claims`, extracts `policy_id`
2. **RAG Agent** — Queries Pinecone with the `narrative` to retrieve the policy; checks coverage and exclusions
3. **History Agent** — Queries `Claim_History` in MongoDB by `customer_id`; flags repeat offenders
4. **Fraud Detection Agent** — Cross-references `ocr_extraction` fields across all active claims; spots collusion signals
5. **Decision Agent** — Combines all signals and produces an outcome: Approve, Deny, Escalate, or Flag for Investigation

The `fraud_scenario` field on each `Active_Claims` record is your ground truth for evaluating whether each agent made the correct call.

---

## Key Libraries Used

| Library | Purpose |
|---|---|
| `pymongo` | Connect to and write records in MongoDB Atlas |
| `pinecone-client` | Connect to Pinecone, create the index, and upsert vectors |
| `sentence-transformers` | Run the `all-MiniLM-L6-v2` model locally to generate 384-dim embeddings |
| `faker` | Generate realistic customer names and company names |
| `python-dotenv` | Load `MONGODB_URL` and `PINECONE_API_KEY` from the `.env` file |

[MongoDB] Inserted 50 Customer_Profiles
[MongoDB] Inserted 98 Claim_History records
[MongoDB] Inserted 50 Active_Claims
[Pinecone] Upserted 50 Insurance_Policies vectors

[Summary] Fraud scenario distribution:
  aggregate_breach: 4 records
  collusion_ring: 8 records
  frequent_claimant: 5 records
  normal: 18 records
  semantic_exclusion: 10 records
  staged_accident: 5 records

[Done] Referential integrity: all policy_id and customer_id values are consistent across MongoDB and Pinecone.