# utils/metrics.py
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)

def evaluate_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob > threshold).astype(int)

    metrics = {
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1-score": f1_score(y_true, y_pred),
        "ROC-AUC": roc_auc_score(y_true, y_prob)
    }

    return metrics, confusion_matrix(y_true, y_pred), classification_report(y_true, y_pred)
