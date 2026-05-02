from motor.motor_asyncio import AsyncIOMotorClient
from app.core.config import settings

# Initialize database connection
client = AsyncIOMotorClient(settings.MONGODB_URI)
db = client[settings.DATABASE_NAME]

async def get_policy_by_number(policy_number: str) -> dict:
    """Fetch policy by its policy_number."""
    if not policy_number:
        return None
    return await db.policies.find_one({"policy_number": policy_number})

async def get_policyholder_by_id(user_id: str) -> dict:
    """Fetch user details by policyholder_id."""
    if not user_id:
        return None
    return await db.policyholders.find_one({"policyholder_id": user_id})

async def get_vehicle_by_id(vehicle_id: str) -> dict:
    """Fetch vehicle details by vehicle_id."""
    if not vehicle_id:
        return None
    return await db.vehicles.find_one({"vehicle_id": vehicle_id})

async def get_claims_by_vehicle(vehicle_id: str) -> list:
    """Fetch past claims associated with a vehicle_id."""
    if not vehicle_id:
        return []
    cursor = db.claims.find({"vehicle_id": vehicle_id})
    return await cursor.to_list(length=100)

async def get_claims_by_policyholder(policyholder_id: str) -> list:
    """Fetch past claims associated with a user."""
    # Find all policies for user, then claims for those policies
    cursor = db.policies.find({"policyholder_id": policyholder_id})
    policies = await cursor.to_list(length=100)
    policy_numbers = [p["policy_number"] for p in policies]
    
    if not policy_numbers:
        return []
        
    claims_cursor = db.claims.find({"policy_number": {"$in": policy_numbers}})
    return await claims_cursor.to_list(length=100)

async def get_fraud_signals_by_entity(entity_id: str) -> list:
    """Fetch any active fraud signals for a user or vehicle."""
    if not entity_id:
        return []
    cursor = db.fraud_signals.find({"entity_id": entity_id})
    return await cursor.to_list(length=100)
