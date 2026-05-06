import os
import uuid
import random
from datetime import datetime, timedelta
from dotenv import load_dotenv
from faker import Faker
from pymongo import MongoClient
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

from pathlib import Path
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")


os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")
fake = Faker()
model = SentenceTransformer("all-MiniLM-L6-v2")

mongo = MongoClient(os.getenv("MONGODB_URL"))
db = mongo["car_insurance_mas"]

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
INDEX_NAME = "insurance-policies"

if INDEX_NAME not in [i.name for i in pc.list_indexes()]:
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
index = pc.Index(INDEX_NAME)

db["Customer_Profiles"].drop()
db["Claim_History"].drop()
db["Active_Claims"].drop()

COLLUSION_SHOP = "Apex AutoBody & Collision"
NUM_CUSTOMERS = 50

COVERAGE_TEMPLATES = [
    "Comprehensive coverage including glass breakage, fire, theft, and third-party bodily injury liability up to policy limit.",
    "Third-party liability only covering property damage and bodily injury to others; own vehicle damage excluded.",
    "Comprehensive coverage including flood, hail, vandalism, animal strike, and uninsured motorist protection.",
    "Collision coverage for damage resulting from impact with another vehicle or stationary object, plus roadside assistance.",
    "Full coverage with GAP insurance, rental reimbursement, and new car replacement within first 24 months.",
]

EXCLUSION_TEMPLATES = [
    "Excludes: Street Racing, DUI/DWI, Off-roading on unpaved terrain, Unlicensed Drivers, Intentional Damage.",
    "Excludes: Racing events, vehicles used for hire (rideshare/taxi), mechanical breakdown, wear and tear.",
    "Excludes: War, nuclear hazard, government seizure, use while intoxicated, illegal modifications.",
    "Excludes: Street Racing, commercial use, driving outside licensed territory, off-road circuits.",
    "Excludes: DUI, Unlicensed Drivers, intentional self-damage, contraband transport, Off-roading.",
]

INCIDENT_TYPES = ["Rear-end", "Theft", "Vandalism", "Hit and Run", "Total Loss"]
CLAIM_STATUSES = ["Approved", "Denied", "Fraud_Flagged"]

NORMAL_NARRATIVES = [
    "While stopped at a red light, the claimant's vehicle was struck from behind by an unidentified pickup truck that fled the scene immediately.",
    "The vehicle was discovered missing from the residential driveway overnight. A police report was filed the following morning.",
    "Claimant reports returning to a parking lot to find deep key scratches running along both driver-side doors and a broken side mirror.",
    "A delivery van ran a stop sign and collided with the front passenger side of the insured vehicle at a suburban intersection.",
    "Severe hailstorm caused multiple dents across the hood, roof, and trunk. Windshield cracked in three locations.",
    "The vehicle skidded on an icy road and impacted a concrete median barrier. Airbags deployed and front bumper is destroyed.",
    "Claimant's parked vehicle was struck by a reversing neighbor who left a note with their contact details.",
    "A stray shopping cart in a grocery store parking lot rolled into the vehicle, causing a dent and paint damage on the rear quarter panel.",
]

FRAUD_NARRATIVES = {
    "frequent_claimant_vandalism": [
        "Claimant reports the vehicle was vandalized overnight; all four tires slashed and 'PAYBACK' keyed into the hood. No witnesses and no camera footage available.",
        "Returned to find the windshield smashed and the interior ransacked. Claims a laptop and camera equipment were also stolen from inside the vehicle.",
    ],
    "staged_accident": [
        "A massive multi-vehicle pileup occurred on the highway involving at least twelve vehicles. Multiple lanes were blocked for hours. Emergency services were on scene for six hours. Claimant's bumper shows a small scuff.",
        "A catastrophic chain-reaction collision during rush hour involving nine vehicles including two semi-trucks. The insured vehicle was supposedly caught in the center of the pile-up. Repair estimate submitted is $200.",
    ],
    "semantic_exclusion": [
        "Claimant was driving at high speed on a closed circuit track during a private event when they lost control on a hairpin and struck the barrier.",
        "Vehicle was damaged while the driver was participating in a timed lap competition on a closed-circuit road course. Engine bay sustained fire damage.",
    ],
    "collusion_ring": [
        "Claimant reports significant front-end damage following a collision at a busy intersection. Vehicle towed directly to Apex AutoBody & Collision for inspection.",
        "The vehicle sustained damage to the rear bumper and trunk. Claimant chose Apex AutoBody & Collision based on a friend's recommendation for repairs.",
        "Hail damage across the vehicle's roof and hood was assessed. Claimant dropped the vehicle off at Apex AutoBody & Collision the following day.",
        "Side-swipe damage along the entire driver side. Vehicle is currently at Apex AutoBody & Collision awaiting parts.",
        "Claimant discovered vandalism damage after leaving a shopping center. Immediately drove to Apex AutoBody & Collision for an estimate.",
    ],
}


def gen_date(days_back_max=730):
    return (datetime.now() - timedelta(days=random.randint(0, days_back_max))).strftime("%Y-%m-%d")


def gen_history(customer_id, num_claims, fraud_heavy=False):
    records = []
    for _ in range(num_claims):
        if fraud_heavy:
            status = random.choice(["Denied", "Fraud_Flagged", "Denied"])
        else:
            status = random.choice(CLAIM_STATUSES)
        records.append({
            "history_id": str(uuid.uuid4()),
            "customer_id": customer_id,
            "incident_type": random.choice(INCIDENT_TYPES),
            "payout_amount": round(random.uniform(500, 18000), 2) if status == "Approved" else 0.0,
            "claim_status": status,
            "incident_date": gen_date(),
        })
    return records


customers = []
all_histories = []
policies_pinecone = []
active_claims = []

frequent_claimant_ids = []
collusion_customer_ids = []

for i in range(NUM_CUSTOMERS):
    cid = str(uuid.uuid4())
    pid = str(uuid.uuid4())

    is_frequent_claimant = i < 5
    is_collusion = 5 <= i < 10
    is_staged = 10 <= i < 15
    is_semantic_exclusion = 15 <= i < 18
    is_aggregate_breach = 18 <= i < 22

    if is_frequent_claimant:
        frequent_claimant_ids.append(cid)
        history = gen_history(cid, num_claims=random.randint(3, 5), fraud_heavy=True)
        risk = "High Risk"
        ncd = 0.0
    elif is_collusion:
        collusion_customer_ids.append(cid)
        history = gen_history(cid, num_claims=random.randint(1, 3))
        risk = random.choice(["Safe", "Watchlist"])
        ncd = round(random.uniform(0.1, 0.4), 2)
    else:
        num_hist = random.randint(0, 4)
        history = gen_history(cid, num_claims=num_hist)
        recent_denied = sum(1 for h in history if h["claim_status"] in ["Denied", "Fraud_Flagged"])
        risk = "High Risk" if recent_denied >= 3 else ("Watchlist" if recent_denied >= 1 else "Safe")
        ncd = round(random.uniform(0.0, 0.5), 2)

    all_histories.extend(history)
    total_paid = sum(h["payout_amount"] for h in history)

    policy_limit = random.choice([15000, 20000, 25000, 30000])
    aggregate_limit = random.choice([50000, 75000, 100000])
    deductible = random.choice([250, 500, 750, 1000])
    coverage = random.choice(COVERAGE_TEMPLATES)
    exclusions = random.choice(EXCLUSION_TEMPLATES)

    customer = {
        "customer_id": cid,
        "full_name": fake.name(),
        "tenure_months": random.randint(1, 240),
        "ncd_tier": ncd,
        "risk_rating": risk,
        "policy_id": pid,
    }
    customers.append(customer)

    policy_text = (
        f"Policy ID: {pid}. Customer: {customer['full_name']}. "
        f"Coverage: {coverage} "
        f"Policy Limit: ${policy_limit:,}. "
        f"Aggregate Limit: ${aggregate_limit:,}. "
        f"Deductible: ${deductible:,}. "
        f"Exclusions: {exclusions}"
    )
    embedding = model.encode(policy_text).tolist()

    policies_pinecone.append({
        "id": pid,
        "values": embedding,
        "metadata": {
            "policy_id": pid,
            "customer_id": cid,
            "customer_name": customer["full_name"],
            "coverage_scope": coverage,
            "policy_limit": policy_limit,
            "aggregate_limit": aggregate_limit,
            "deductible": deductible,
            "exclusions": exclusions,
            "total_historical_payout": total_paid,
        },
    })

    if is_frequent_claimant:
        narrative = random.choice(FRAUD_NARRATIVES["frequent_claimant_vandalism"])
        incident_type = "Vandalism"
        estimated_loss = round(random.uniform(800, 3000), 2)
        shop = fake.company() + " Auto Repair"
    elif is_collusion:
        narrative = FRAUD_NARRATIVES["collusion_ring"][i - 5]
        incident_type = random.choice(["Rear-end", "Vandalism", "Hit and Run"])
        estimated_loss = round(random.uniform(2000, 12000), 2)
        shop = COLLUSION_SHOP
    elif is_staged:
        narrative = random.choice(FRAUD_NARRATIVES["staged_accident"])
        incident_type = "Rear-end"
        estimated_loss = round(random.uniform(8000, 20000), 2)
        shop = fake.company() + " Collision Center"
        ocr_estimate = 200.0
    elif is_semantic_exclusion:
        narrative = FRAUD_NARRATIVES["semantic_exclusion"][i - 15]
        incident_type = "Total Loss"
        estimated_loss = round(random.uniform(15000, 28000), 2)
        shop = fake.company() + " Motorsport Repairs"
    elif is_aggregate_breach:
        breach_amount = aggregate_limit - total_paid + random.uniform(5000, 15000)
        estimated_loss = round(max(breach_amount, 1000), 2)
        narrative = random.choice(NORMAL_NARRATIVES)
        incident_type = random.choice(INCIDENT_TYPES)
        shop = fake.company() + " Auto Body"
    else:
        narrative = random.choice(NORMAL_NARRATIVES)
        incident_type = random.choice(INCIDENT_TYPES)
        estimated_loss = round(random.uniform(500, policy_limit * 0.9), 2)
        shop = fake.company() + " Auto Repair"

    loss_date = gen_date(days_back_max=180)

    ocr_extraction = {
        "PolicyNumber": pid[:8].upper(),
        "ClaimantName": customer["full_name"],
        "LossDate": loss_date,
        "RepairShopName": shop,
        "TotalEstimate": ocr_estimate if is_staged else estimated_loss,
    }

    active_claims.append({
        "claim_id": str(uuid.uuid4()),
        "policy_id": pid,
        "customer_id": cid,
        "incident_type": incident_type,
        "incident_date": loss_date,
        "narrative": narrative,
        "ocr_extraction": ocr_extraction,
        "estimated_loss": estimated_loss,
        "fraud_scenario": (
            "frequent_claimant" if is_frequent_claimant else
            "collusion_ring" if is_collusion else
            "staged_accident" if is_staged else
            "semantic_exclusion" if is_semantic_exclusion else
            "aggregate_breach" if is_aggregate_breach else
            "normal"
        ),
    })

db["Customer_Profiles"].insert_many(customers)
print(f"[MongoDB] Inserted {len(customers)} Customer_Profiles")

db["Claim_History"].insert_many(all_histories)
print(f"[MongoDB] Inserted {len(all_histories)} Claim_History records")

db["Active_Claims"].insert_many(active_claims)
print(f"[MongoDB] Inserted {len(active_claims)} Active_Claims")

BATCH_SIZE = 50
for start in range(0, len(policies_pinecone), BATCH_SIZE):
    batch = policies_pinecone[start: start + BATCH_SIZE]
    index.upsert(vectors=batch)
print(f"[Pinecone] Upserted {len(policies_pinecone)} Insurance_Policies vectors")

db["Customer_Profiles"].create_index("customer_id", unique=True)
db["Claim_History"].create_index("customer_id")
db["Claim_History"].create_index("history_id", unique=True)
db["Active_Claims"].create_index("claim_id", unique=True)
db["Active_Claims"].create_index("policy_id")

print("\n[Summary] Fraud scenario distribution:")
from collections import Counter
scenario_counts = Counter(c["fraud_scenario"] for c in active_claims)
for scenario, count in scenario_counts.items():
    print(f"  {scenario}: {count} records")

print("\n[Done] Referential integrity: all policy_id and customer_id values are consistent across MongoDB and Pinecone.")