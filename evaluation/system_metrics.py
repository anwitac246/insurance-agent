import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def calculate_system_metrics(results_df):
    total_claims = len(results_df)
    if total_claims == 0:
        return {}
        
    correct_decisions = (results_df['true_decision'] == results_df['pred_decision']).sum()
    automation_rate = (results_df['pred_decision'] != 'escalate').sum() / total_claims
    escalation_rate = (results_df['pred_decision'] == 'escalate').sum() / total_claims
    
    # Regression metrics for payout
    valid_payouts = results_df[results_df['true_payout'] > 0]
    if not valid_payouts.empty:
        mae = mean_absolute_error(valid_payouts['true_payout'], valid_payouts['pred_payout'])
        rmse = np.sqrt(mean_squared_error(valid_payouts['true_payout'], valid_payouts['pred_payout']))
    else:
        mae, rmse = 0, 0
        
    avg_latency = results_df['latency'].mean()
    
    return {
        "overall_accuracy": float(correct_decisions / total_claims),
        "automation_rate": float(automation_rate),
        "escalation_rate": float(escalation_rate),
        "error_rate": float(1 - (correct_decisions / total_claims)),
        "payout_mae": float(mae),
        "payout_rmse": float(rmse),
        "avg_latency_seconds": float(avg_latency)
    }
