import json
import random
import os
from faker import Faker

fake = Faker()

def generate_dataset(num_samples=100, output_path="data/eval_dataset.json"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    dataset = []
    
    # 40% Valid, 30% Fraud, 20% Missing, 10% Edge Cases
    counts = {
        "valid": int(num_samples * 0.40),
        "fraud": int(num_samples * 0.30),
        "missing": int(num_samples * 0.20),
        "edge": int(num_samples * 0.10)
    }
    
    # Fix rounding if it doesn't sum to exactly num_samples
    while sum(counts.values()) < num_samples:
        counts["valid"] += 1

    claim_id_counter = 1
    
    for scenario, count in counts.items():
        for _ in range(count):
            claim_id = f"EVAL-{claim_id_counter:04d}"
            policy_num = f"POL-{fake.unique.random_number(digits=6)}"
            veh_num = fake.license_plate()
            user_name = fake.name()
            
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
            
            base_text = f"Policy Number: {policy_num}\nName: {user_name}\nVehicle: {veh_num}\nDate: 2023-10-01\n"
            
            if scenario == "valid":
                case["input"]["extracted_text"] = base_text + "FIR Number: FIR-123\nRepair Estimate: $1500\nSignature: YES\nIdentity: PASSPORT-999"
                case["ground_truth"]["decision"] = "approve"
                case["ground_truth"]["expected_payout"] = 1500.0
                
            elif scenario == "fraud":
                case["input"]["extracted_text"] = base_text + "FIR Number: FIR-FAKE\nRepair Estimate: $8500\nSignature: YES\nIdentity: PASSPORT-999"
                case["ground_truth"]["decision"] = "reject"
                case["ground_truth"]["is_fraud"] = True
                
            elif scenario == "missing":
                # Miss the repair estimate
                case["input"]["extracted_text"] = base_text + "FIR Number: FIR-123\nSignature: YES\nIdentity: PASSPORT-999"
                case["ground_truth"]["decision"] = "pending_docs"
                case["ground_truth"]["missing_docs"] = ["Repair Estimate"]
                
            elif scenario == "edge":
                # e.g., expired policy or extremely blurry OCR
                case["input"]["extracted_text"] = "P0l!cy Numb3r: @#$\nN4me: J()hn D0e" # OCR noise
                case["input"]["images_present"] = False
                case["ground_truth"]["decision"] = "escalate"
                case["ground_truth"]["policy_valid"] = False
                case["ground_truth"]["missing_fields"] = ["policy.policy_number"]
            
            dataset.append(case)
            claim_id_counter += 1
            
    # Shuffle dataset
    random.shuffle(dataset)
            
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4)
        
    print(f"Generated {len(dataset)} evaluation cases at {output_path}")

if __name__ == "__main__":
    generate_dataset(100)
