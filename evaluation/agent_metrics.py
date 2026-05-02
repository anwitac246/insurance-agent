from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def calculate_fraud_metrics(y_true, y_pred):
    if not y_true: return {}
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred))
    }

def calculate_policy_metrics(y_true_valid, y_pred_valid):
    if not y_true_valid: return {}
    return {
        "accuracy": float(accuracy_score(y_true_valid, y_pred_valid))
    }
    
def calculate_document_metrics(true_missing_docs, pred_missing_docs):
    false_missing = 0
    correct_missing = 0
    total = len(true_missing_docs)
    
    for t, p in zip(true_missing_docs, pred_missing_docs):
        # Using simple set matching for list of missing docs
        set_t = set(t)
        set_p = set(p)
        if len(set_p - set_t) > 0:
            false_missing += 1
        if set_p == set_t:
            correct_missing += 1
            
    return {
        "missing_doc_accuracy": float(correct_missing / total) if total > 0 else 0.0,
        "false_missing_rate": float(false_missing / total) if total > 0 else 0.0
    }
