# Car Insurance Test Data Seeding Guide (`scripts/`)

This directory contains `seed_data.py`, which generates **50 interconnected, synthetic records** across two databases to act as the Ground Truth dataset for the evaluating the agents.

## Data Schema
Four collections/indexes are created to mirror a real insurance backend:

```
MongoDB
├── Customer_Profiles      — Core profile, tenure, and overall risk rating
├── Claim_History          — Past claims and their outcomes (Approved/Denied)
└── Active_Claims          — The 50 new claims the agents will process today

Pinecone
└── insurance-policies     — One policy document per customer, vectorized
```

## How the IDs Connect
Every record is tightly linked by two UUIDs generated per customer:
- `customer_id` connects Profiles, History, Active Claims, and Pinecone metadata.
- `policy_id` connects Active Claims, Profiles, and the actual Pinecone vector.
This guarantees 100% referential integrity—no orphaned records.

## Data Generation Details

### Customer Profiles & Claim History
50 customers are created using `faker`. They are assigned a tenure, a no-claim discount (NCD) tier, and a randomly sized batch of past claims. Fraud-heavy profiles receive histories loaded with `Denied` and `Fraud_Flagged` statuses.

### Insurance Policies (Pinecone)
A plain English string is assembled defining the customer's coverage limit, aggregate limit, deductible, and exclusions (e.g., "Excludes: Street Racing, DUI, Off-roading"). 
This string is vectorized using `sentence-transformers` (`all-MiniLM-L6-v2`) and upserted to Pinecone. 

### Active Claims
One active claim per customer containing:
- `narrative`: A paragraph describing the incident.
- `ocr_extraction`: Extracted fields like repair shop name and estimate.
- `estimated_loss`: The requested payout.
- `fraud_scenario`: A hidden label used to grade the agent's performance.

## Fraud & Edge Case Scenarios
40% of the active claims are specifically crafted to test agent reasoning:

1. **The Frequent False Claimant (Records 0–4)**
   - **Data**: The customer has 3+ denied/flagged historical claims and is submitting a new Vandalism claim.
   - **Tests**: Historical cross-referencing.

2. **The Collusion Ring (Records 5–9)**
   - **Data**: Five unrelated customers all use `Apex AutoBody & Collision`.
   - **Tests**: Cross-claim analysis.

3. **The Staged Accident (Records 10–14)**
   - **Data**: Narrative describes a catastrophic multi-car pileup, but the OCR repair estimate is only $200.
   - **Tests**: Narrative-to-Evidence mismatch detection.

4. **The Semantic Exclusion (Records 15–17)**
   - **Data**: Narrative describes a "closed circuit track event". Policy excludes "Street Racing".
   - **Tests**: Semantic matching via Vector DB (not just keyword matching).

5. **The Aggregate Limit Breach (Records 18–21)**
   - **Data**: The active `estimated_loss` plus historical payouts exceeds the policy's lifetime aggregate limit.
   - **Tests**: Complex financial arithmetic.

6. **Normal Claims (Remaining Records)**
   - **Data**: Legitimate claims under policy limits.
   - **Tests**: Ensuring the agent doesn't over-escalate or hallucinate fraud.

## Running the Script

Create a `.env` file in the root directory:
```env
MONGODB_URL=mongodb+srv://<user>:<password>@<cluster>.mongodb.net/
PINECONE_API_KEY=your-pinecone-api-key
```

Run the seeder:
```bash
python scripts/seed_data.py
```
*Note: The script drops and recreates the MongoDB collections every time it runs to ensure a clean test state.*