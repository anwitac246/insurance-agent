import json
import random
import os
import asyncio
from faker import Faker
from motor.motor_asyncio import AsyncIOMotorClient

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.core.config import settings

fake = Faker()

async def fetch_real_data_from_db(limit=200):
    client = AsyncIOMotorClient(settings.MONGODB_URI)
    db = client[settings.DATABASE_NAME]
    
    policies_cursor = db.policies.aggregate([
        {"$lookup": {"from": "policyholders", "localField": "policyholder_id", "foreignField": "policyholder_id", "as": "user"}},
        {"$lookup": {"from": "vehicles", "localField": "vehicle_id", "foreignField": "vehicle_id", "as": "vehicle"}},
        {"$match": {"user": {"$ne": []}, "vehicle": {"$ne": []}}},
        {"$limit": limit}
    ])
    
    real_data = []
    async for p in policies_cursor:
        real_data.append({
            "policy_num": p["policy_number"],
            "veh_num": p["vehicle"][0]["vehicle_number"],
            "user_name": p["user"][0]["name"]
        })
        
    client.close()
    return real_data

async def generate_dataset(num_samples=100, output_path="data/eval_dataset.json"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    dataset = []
    
    counts = {
        "valid": int(num_samples * 0.40),
        "fraud": int(num_samples * 0.30),
        "missing": int(num_samples * 0.20),
        "edge": int(num_samples * 0.10)
    }
    
    while sum(counts.values()) < num_samples:
        counts["valid"] += 1

    claim_id_counter = 1
    
    # Fetch real linked data from MongoDB
    print("Fetching real entities from MongoDB...")
    real_data_pool = await fetch_real_data_from_db(limit=num_samples + 50)
    if not real_data_pool:
         print("Warning: MongoDB is empty! Please run generate_synthetic_db.py first. Falling back to random faker data.")
         # Fallback pool
         real_data_pool = [{"policy_num": f"POL-{fake.unique.random_number(digits=6)}", "veh_num": fake.license_plate(), "user_name": fake.name()} for _ in range(num_samples)]
         
    random.shuffle(real_data_pool)
    
    for scenario, count in counts.items():
        for _ in range(count):
            if not real_data_pool:
                break
                
            db_item = real_data_pool.pop()
            
            claim_id = f"EVAL-{claim_id_counter:04d}"
            policy_num = db_item["policy_num"]
            veh_num = db_item["veh_num"]
            user_name = db_item["user_name"]
            
            case = {
                "claim_id": claim_id,
                "scenario_type": scenario,
                "input": {
                    "extracted_text": "",
                    "images_present": True,
                },
                "ground_truth": {
                    "decision": "",
                    "is_fraud": False,
                    "policy_valid": True,
                    "missing_docs": [],
                    "missing_fields": [],
                    "expected_payout": 0.0
                }
            }
            
            fir_num = fake.random_number(digits=5)
            # Master template containing all 8 documents explicitly formatted
            base_text = f"""
=== INSURANCE POLICY ===
Policy Number: {policy_num}
Policyholder Name: {user_name}
Start Date: 2022-01-01
End Date: 2025-01-01
Type: Comprehensive
IDV: 15000.0
Vehicle Number: {veh_num}
Coverage: Full Collision
Add-ons: Zero Depreciation

=== CLAIM FORM ===
Claim ID: {claim_id}
Policy Number: {policy_num}
Claimant Name: {user_name}
Incident Date: 2023-10-01
Incident Time: 14:30
Description: Rear-ended at a stoplight.
Location: Main St Intersection
Signature Present: True

=== VEHICLE RC ===
Vehicle Number: {veh_num}
Owner Name: {user_name}
Registration Date: 2020-05-10
Chassis Number: VIN-{fake.random_number(digits=8)}
Engine Number: ENG-{fake.random_number(digits=8)}
Make Model: Toyota Corolla

=== DRIVING LICENSE ===
Driver Name: {user_name}
License Number: DL-{fake.random_number(digits=8)}
Issue Date: 2018-01-01
Expiry Date: 2028-01-01
Class: LMV

=== FIR DOCUMENT ===
FIR Number: FIR-{fir_num}
FIR Date: 2023-10-02
Description: Two-car collision.
Police Station: Central Precinct

=== DAMAGE IMAGES ===
Damage Summary: Heavy rear bumper damage.
Regions: Rear Bumper, Trunk
Severity: High
Quality Score: 0.9
Is Blurry: False

=== REPAIR ESTIMATE ===
Garage Name: Joe's Auto
Estimated Amount: 2500.0
Damage Details: Bumper replacement and trunk realignment.
Invoice Number: INV-999

=== IDENTITY PROOF ===
Name: {user_name}
ID Number: PASS-{fake.random_number(digits=6)}
Address: 123 Main St, Anytown
"""
            
            if scenario == "valid":
                case["input"]["extracted_text"] = base_text
                case["ground_truth"]["decision"] = "approve"
                case["ground_truth"]["expected_payout"] = 2500.0
                
            elif scenario == "fraud":
                # Inject a mismatch to trigger fraud
                fraud_text = base_text.replace(f"Vehicle Number: {veh_num}", "Vehicle Number: FAKE-999")
                fraud_text = fraud_text.replace("Estimated Amount: 2500.0", "Estimated Amount: 15000.0")
                case["input"]["extracted_text"] = fraud_text
                case["ground_truth"]["decision"] = "reject"
                case["ground_truth"]["is_fraud"] = True
                
            elif scenario == "missing":
                # Remove the entire FIR section
                missing_text = base_text.replace(f"""=== FIR DOCUMENT ===
FIR Number: FIR-{fir_num}
FIR Date: 2023-10-02
Description: Two-car collision.
Police Station: Central Precinct""", "")
                case["input"]["extracted_text"] = missing_text
                case["ground_truth"]["decision"] = "pending_docs"
                case["ground_truth"]["missing_docs"] = ["FIRDocument"]
                
            elif scenario == "edge":
                # Corrupt the text
                corrupted = base_text.replace("0", "O").replace("1", "I").replace("e", "3")
                case["input"]["extracted_text"] = corrupted
                case["input"]["images_present"] = False
                case["ground_truth"]["decision"] = "escalate"
                case["ground_truth"]["policy_valid"] = False # Edge case corrupts policy number
                case["ground_truth"]["missing_fields"] = ["policy.policy_number"]
            
            dataset.append(case)
            claim_id_counter += 1
            
    random.shuffle(dataset)
            
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4)
        
    print(f"Generated {len(dataset)} evaluation cases at {output_path}")

if __name__ == "__main__":
    asyncio.run(generate_dataset(100))
