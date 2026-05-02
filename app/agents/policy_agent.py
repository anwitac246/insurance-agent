from app.models.state import ClaimState
from app.services.db_queries import get_policy_by_number, get_policyholder_by_id, get_vehicle_by_id
import asyncio
from datetime import datetime

async def verify_policy(state: ClaimState) -> ClaimState:
    extracted = state.get("extracted_data", {})
    policy_doc = extracted.get("insurance_policy", {})
    claim_form = extracted.get("claim_form", {})
    
    # Fallback to claim_form policy number if not found in policy_doc
    policy_num = (policy_doc.get("policy_number") or claim_form.get("policy_number") or "").strip()
    
    async def run_queries():
        policy = await get_policy_by_number(policy_num) if policy_num else None
        user = None
        vehicle = None
        if policy:
            user = await get_policyholder_by_id(policy.get("policyholder_id"))
            vehicle = await get_vehicle_by_id(policy.get("vehicle_id"))
        return policy, user, vehicle
        
    try:
        policy, user, vehicle = await run_queries()
        
        if not policy:
            state["policy_verification"] = {
                "policy_valid": False,
                "user_match": False,
                "vehicle_match": False,
                "within_policy_period": False,
                "details": f"Policy {policy_num} not found in internal database."
            }
            return state

        # Validate
        is_active = policy.get("status") == "active"
        
        incident_date_str = claim_form.get("incident_date")
        within_period = False
        if incident_date_str:
            try:
                incident_date = datetime.fromisoformat(incident_date_str).date()
                start = datetime.fromisoformat(policy["start_date"]).date()
                end = datetime.fromisoformat(policy["end_date"]).date()
                within_period = start <= incident_date <= end
            except Exception:
                pass
                
        extracted_name = (policy_doc.get("policyholder_name") or claim_form.get("claimant_name") or "").lower().strip()
        db_name = (user.get("name") or "").lower().strip() if user else ""
        user_match = (extracted_name == db_name) if extracted_name and db_name else False
        
        extracted_veh = (policy_doc.get("vehicle_number") or "").replace(" ", "").replace("-", "").upper()
        db_veh = (vehicle.get("vehicle_number") or "").replace(" ", "").replace("-", "").upper() if vehicle else ""
        vehicle_match = (extracted_veh == db_veh) if extracted_veh and db_veh else False
        
        state["policy_verification"] = {
            "policy_valid": is_active,
            "user_match": user_match,
            "vehicle_match": vehicle_match,
            "within_policy_period": within_period,
            "details": f"Policy found. Status: {policy['status']}. Vehicle: {db_veh}. User: {db_name}."
        }
        
    except Exception as e:
        state["policy_verification"] = {
            "policy_valid": False,
            "user_match": False,
            "vehicle_match": False,
            "within_policy_period": False,
            "details": f"Database verification failed: {str(e)}"
        }
        
    return state
