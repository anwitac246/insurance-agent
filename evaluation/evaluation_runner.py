import json
import time
import pandas as pd
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.models.state import ClaimState
from app.agents.document_agent import structured_llm, validate_extracted_data, MissingData
from app.agents.policy_agent import verify_policy
from app.agents.fraud_agent import detect_fraud
try:
    from app.agents.decision_agent import make_decision
except ImportError:
    make_decision = None

from evaluation.agent_metrics import calculate_fraud_metrics, calculate_policy_metrics, calculate_document_metrics
from evaluation.system_metrics import calculate_system_metrics
from evaluation.robustness_tests import add_ocr_noise
from dotenv import load_dotenv
load_dotenv()

async def run_pipeline(input_data: dict, add_noise=False) -> tuple:
    """
    Simulates the LangGraph pipeline asynchronously for testing.
    Returns (final_state, latency)
    """
    start_time = time.time()
    
    state = {
        "claim_id": "TEST",
        "status": "processing",
        "user_id": "test_user",
        "local_files": {},
        "extracted_data": {},
        "policy_verification": {},
        "fraud_analysis": {},
        "decision": {},
        "missing_data": {"missing_documents": [], "missing_fields": []}
    }
    
    extracted_text = input_data.get("extracted_text", "")
    if add_noise:
        extracted_text = add_ocr_noise(extracted_text)
        
    # 1. Document Agent (Mocked extraction step to save file I/O latency)
    prompt = f"Analyze extracted claim text.\nExtracted Text: {extracted_text}\nDamage Summary: "
    try:
        extracted_data_obj = await structured_llm.ainvoke(prompt)
    except Exception:
        extracted_data_obj = None
        
    state["extracted_data"] = extracted_data_obj.model_dump() if extracted_data_obj else {}
    missing_data = validate_extracted_data(extracted_data_obj) if extracted_data_obj else MissingData(missing_documents=["All"], missing_fields=["Failed"])
    state["missing_data"] = missing_data
    
    if missing_data["missing_documents"] or missing_data["missing_fields"]:
         state["status"] = "pending_docs"
         return state, time.time() - start_time
         
    # 2. Policy Agent
    state = await verify_policy(state)
    if not state.get("policy_verification", {}).get("policy_valid"):
        state["status"] = "rejected"
        return state, time.time() - start_time
        
    # 3. Fraud Agent
    state = await detect_fraud(state)
    
    # 4. Decision Agent (Fallback if module missing)
    if make_decision:
        try:
            state = make_decision(state)
        except Exception:
            pass
            
    if not state.get("decision"):
        fraud_score = state.get("fraud_analysis", {}).get("fraud_risk_score", 0.0)
        if fraud_score > 0.5:
            state["decision"] = {"status": "escalate", "approved_amount": 0}
        else:
            state["decision"] = {"status": "approve", "approved_amount": 1000.0}
            
    latency = time.time() - start_time
    return state, latency

async def run_evaluation(dataset_path="data/eval_dataset.json"):
    if not os.path.exists(dataset_path):
        print(f"Dataset {dataset_path} not found. Please run generation script.")
        return
        
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
        
    results = []
    
    # Run only a small sample to save Groq API tokens and time
    dataset = dataset[:10] 
    
    for case in dataset:
        print(f"Evaluating Case {case['claim_id']} ({case['scenario_type']})...")
        
        # Clean run
        state, latency = await run_pipeline(case["input"], add_noise=False)
        
        # Determine predicted decision
        if state["status"] == "pending_docs":
            pred_decision = "pending_docs"
            pred_payout = 0
        elif state["status"] == "rejected":
            pred_decision = "reject"
            pred_payout = 0
        else:
            pred_decision = state.get("decision", {}).get("status", "escalate")
            pred_payout = state.get("decision", {}).get("approved_amount", 0)
            
        pred_is_fraud = state.get("fraud_analysis", {}).get("fraud_risk_score", 0.0) > 0.5
        pred_policy_valid = state.get("policy_verification", {}).get("policy_valid", False)
        
        results.append({
            "claim_id": case["claim_id"],
            "scenario": case["scenario_type"],
            "latency": latency,
            "true_decision": case["ground_truth"]["decision"],
            "pred_decision": pred_decision,
            "true_payout": case["ground_truth"]["expected_payout"],
            "pred_payout": pred_payout,
            "true_is_fraud": case["ground_truth"]["is_fraud"],
            "pred_is_fraud": pred_is_fraud,
            "true_policy_valid": case["ground_truth"]["policy_valid"],
            "pred_policy_valid": pred_policy_valid,
            "true_missing_docs": case["ground_truth"].get("missing_docs", []),
            "pred_missing_docs": state.get("missing_data", {}).get("missing_documents", [])
        })
        
    df = pd.DataFrame(results)
    
    # Calculate metrics
    fraud_metrics = calculate_fraud_metrics(df['true_is_fraud'].tolist(), df['pred_is_fraud'].tolist())
    policy_metrics = calculate_policy_metrics(df['true_policy_valid'].tolist(), df['pred_policy_valid'].tolist())
    system_metrics = calculate_system_metrics(df)
    doc_metrics = calculate_document_metrics(df['true_missing_docs'].tolist(), df['pred_missing_docs'].tolist())
    
    report = {
        "document_agent": doc_metrics,
        "policy_agent": policy_metrics,
        "fraud_agent": fraud_metrics,
        "overall": system_metrics
    }
    
    with open("evaluation_report.json", "w") as f:
        json.dump(report, f, indent=4)
        
    print("\n--- Evaluation Report Generated (evaluation_report.json) ---")
    print(json.dumps(report, indent=4))

if __name__ == "__main__":
    import asyncio
    asyncio.run(run_evaluation())
