import asyncio
import random
from datetime import datetime, timedelta
from faker import Faker
from motor.motor_asyncio import AsyncIOMotorClient
import os
from dotenv import load_dotenv

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

MONGO_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DATABASE_NAME", "insurance_claims")

fake = Faker()

async def generate_db():
    print(f"Connecting to MongoDB at {MONGO_URI}...")
    client = AsyncIOMotorClient(MONGO_URI)
    db = client[DB_NAME]
    
    # Drop existing collections to start fresh
    print("Dropping existing collections...")
    for coll in ["policyholders", "vehicles", "policies", "claims", "fraud_signals"]:
        await db[coll].drop()

    # 1. Generate Policyholders (1000)
    print("Generating 1000 policyholders...")
    policyholders = []
    for _ in range(1000):
        ph = {
            "policyholder_id": fake.uuid4(),
            "name": fake.name(),
            "contact_number": fake.phone_number(),
            "email": fake.email(),
            "address": fake.address(),
            "kyc_verified": random.random() > 0.05 # 95% verified
        }
        policyholders.append(ph)
    await db.policyholders.insert_many(policyholders)

    # 2. Generate Vehicles (1500)
    print("Generating 1500 vehicles...")
    vehicles = []
    makes = ["Toyota", "Honda", "Ford", "Chevrolet", "Nissan", "Hyundai", "BMW", "Audi"]
    for i in range(1500):
        # 90% owned by existing policyholders, 10% unlinked (edge cases)
        owner = random.choice(policyholders) if random.random() > 0.1 else None
        v = {
            "vehicle_id": fake.uuid4(),
            "vehicle_number": fake.license_plate().replace(" ", "").replace("-", "").upper(),
            "owner_id": owner["policyholder_id"] if owner else fake.uuid4(),
            "make": random.choice(makes),
            "model": fake.word().capitalize(),
            "engine_number": fake.bothify(text='ENG-########?'),
            "chassis_number": fake.bothify(text='VIN-##############?')
        }
        vehicles.append(v)
    await db.vehicles.insert_many(vehicles)

    # 3. Generate Policies (2000)
    print("Generating 2000 policies...")
    policies = []
    for _ in range(2000):
        ph = random.choice(policyholders)
        v = random.choice(vehicles)
        
        start_date = fake.date_between(start_date='-3y', end_date='+1m')
        end_date = start_date + timedelta(days=365)
        
        now = datetime.now().date()
        if start_date <= now <= end_date:
            status = "active"
        elif now > end_date:
            status = "expired"
        else:
            status = "cancelled"
            
        p = {
            "policy_number": f"POL-{fake.unique.random_number(digits=8)}",
            "policyholder_id": ph["policyholder_id"],
            "vehicle_id": v["vehicle_id"],
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "status": status,
            "coverage": random.choice(["comprehensive", "third-party"]),
            "add_ons": random.sample(["zero-depreciation", "engine-protect", "consumables"], k=random.randint(0, 2))
        }
        policies.append(p)
    await db.policies.insert_many(policies)

    # 4. Generate Claims (3000)
    print("Generating 3000 claims...")
    claims = []
    for _ in range(3000):
        pol = random.choice(policies)
        
        # 10% claims fall outside policy period (fraud/error)
        if random.random() > 0.1:
            incident_date = fake.date_between_dates(date_start=datetime.fromisoformat(pol["start_date"]), date_end=datetime.fromisoformat(pol["end_date"]))
        else:
            incident_date = fake.date_between(start_date='-5y', end_date='today')

        c = {
            "claim_id": f"CLM-{fake.unique.random_number(digits=8)}",
            "policy_number": pol["policy_number"],
            "vehicle_id": pol["vehicle_id"],
            "incident_date": incident_date.isoformat(),
            "claim_status": random.choice(["pending", "approved", "rejected", "under_investigation"]),
            "claim_amount": round(random.uniform(500, 25000), 2),
            "fraud_flag": random.random() < 0.05 # 5% naturally flagged
        }
        claims.append(c)
    await db.claims.insert_many(claims)

    # 5. Generate Fraud Signals
    print("Generating Fraud Signals...")
    signals = []
    fraud_ph = random.sample(policyholders, 50)
    for ph in fraud_ph:
        signals.append({
            "entity_id": ph["policyholder_id"],
            "entity_type": "user",
            "fraud_flag": True,
            "fraud_type": random.choice(["identity_theft_suspected", "frequent_claims", "staged_accident_ring"]),
            "risk_score": round(random.uniform(0.7, 1.0), 2)
        })
        
    fraud_v = random.sample(vehicles, 50)
    for v in fraud_v:
        signals.append({
            "entity_id": v["vehicle_id"],
            "entity_type": "vehicle",
            "fraud_flag": True,
            "fraud_type": random.choice(["salvage_title", "vin_cloning_suspected", "previous_total_loss"]),
            "risk_score": round(random.uniform(0.7, 1.0), 2)
        })
    await db.fraud_signals.insert_many(signals)
    
    # Add an explicit test case for predictable testing
    print("Adding explicit test cases...")
    await db.policyholders.insert_one({
        "policyholder_id": "test_user_1",
        "name": "John Doe",
        "contact_number": "555-1234",
        "email": "john.doe@example.com",
        "address": "123 Test St",
        "kyc_verified": True
    })
    
    await db.vehicles.insert_one({
        "vehicle_id": "test_veh_1",
        "vehicle_number": "ABC123", # standard format without hyphens
        "owner_id": "test_user_1",
        "make": "Toyota",
        "model": "Corolla",
        "engine_number": "ENG-111",
        "chassis_number": "VIN-111"
    })
    
    await db.policies.insert_one({
        "policy_number": "POL-9999",
        "policyholder_id": "test_user_1",
        "vehicle_id": "test_veh_1",
        "start_date": (datetime.now() - timedelta(days=100)).date().isoformat(),
        "end_date": (datetime.now() + timedelta(days=265)).date().isoformat(),
        "status": "active",
        "coverage": "comprehensive",
        "add_ons": []
    })

    # Create indexes
    print("Creating indexes...")
    await db.policies.create_index("policy_number", unique=True)
    await db.policyholders.create_index("policyholder_id", unique=True)
    await db.vehicles.create_index("vehicle_id", unique=True)
    await db.claims.create_index("vehicle_id")
    await db.claims.create_index("policy_number")
    await db.fraud_signals.create_index("entity_id")

    print("Database seeding and indexing complete!")
    client.close()

if __name__ == "__main__":
    asyncio.run(generate_db())
