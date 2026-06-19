import numpy as np
import json
from typing import Dict, Tuple
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent))
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, matthews_corrcoef, confusion_matrix
)
from config import (
    COST_FALSE_NEGATIVE_HEART, COST_FALSE_POSITIVE_HEART, REWARD_TRUE_POSITIVE_HEART,
    COST_FALSE_NEGATIVE_DIABETES, COST_FALSE_POSITIVE_DIABETES, REWARD_TRUE_POSITIVE_DIABETES,
    MODEL_REPORT_PATH
)
from logger import get_logger

logger = get_logger("evaluator")


class HealthEvaluator:

    def business_cost(self, y_true, y_pred, fn_cost, fp_cost, tp_reward) -> Tuple:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        cost = (tp * tp_reward) + (fn * -fn_cost) + (fp * -fp_cost)
        return cost, tn, fp, fn, tp

    def optimal_threshold(self, y_true, y_prob, fn_cost, fp_cost, tp_reward) -> float:
        best_t, best_cost = 0.5, -np.inf
        for t in np.arange(0.01, 0.99, 0.01):
            y_pred = (y_prob >= t).astype(int)
            if y_pred.sum() == 0:
                continue
            cost, *_ = self.business_cost(y_true, y_pred, fn_cost, fp_cost, tp_reward)
            if cost > best_cost:
                best_cost = cost
                best_t = t
        return best_t

    def evaluate(self, name: str, y_true, y_prob, fn_cost, fp_cost, tp_reward) -> Dict:
        threshold = self.optimal_threshold(y_true, y_prob, fn_cost, fp_cost, tp_reward)
        y_pred = (y_prob >= threshold).astype(int)
        cost, tn, fp, fn, tp = self.business_cost(y_true, y_pred, fn_cost, fp_cost, tp_reward)
        result = {
            "name": name,
            "threshold": round(float(threshold), 4),
            "auc_roc": round(float(roc_auc_score(y_true, y_prob)), 6),
            "avg_precision": round(float(average_precision_score(y_true, y_prob)), 6),
            "f1": round(float(f1_score(y_true, y_pred)), 6),
            "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 6),
            "recall": round(float(recall_score(y_true, y_pred)), 6),
            "mcc": round(float(matthews_corrcoef(y_true, y_pred)), 6),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            "business_value": int(cost), "y_prob": y_prob
        }
        logger.info(f"{name} | AUC={result['auc_roc']} | Recall={result['recall']} | Value=${cost:,}")
        return result

    def save_report(self, heart_best: Dict, diabetes_best: Dict,
                    heart_features: int, diabetes_features: int,
                    heart_test: int, diabetes_test: int):
        report = {
            "best_heart_model": heart_best["name"],
            "heart_auc_roc": heart_best["auc_roc"],
            "heart_threshold": heart_best["threshold"],
            "heart_f1": heart_best["f1"],
            "heart_recall": heart_best["recall"],
            "heart_business_value": heart_best["business_value"],
            "best_diabetes_model": diabetes_best["name"],
            "diabetes_auc_roc": diabetes_best["auc_roc"],
            "diabetes_threshold": diabetes_best["threshold"],
            "diabetes_f1": diabetes_best["f1"],
            "diabetes_recall": diabetes_best["recall"],
            "diabetes_business_value": diabetes_best["business_value"],
            "heart_feature_count": heart_features,
            "diabetes_feature_count": diabetes_features,
            "heart_test_size": heart_test,
            "diabetes_test_size": diabetes_test
        }
        with open(MODEL_REPORT_PATH, "w") as f:
            json.dump(report, f, indent=4)
        logger.info(f"Report saved to {MODEL_REPORT_PATH}")
        return report