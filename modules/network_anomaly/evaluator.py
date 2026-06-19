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
from config import COST_FALSE_NEGATIVE, COST_FALSE_POSITIVE, REWARD_TRUE_POSITIVE, MODEL_REPORT_PATH
from logger import get_logger

logger = get_logger("evaluator")


class NetworkEvaluator:

    def business_cost(self, y_true, y_pred) -> Tuple:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        cost = int(tp)*REWARD_TRUE_POSITIVE + int(fn)*-COST_FALSE_NEGATIVE + int(fp)*-COST_FALSE_POSITIVE
        return int(cost), int(tn), int(fp), int(fn), int(tp)

    def optimal_threshold(self, y_true, y_prob) -> float:
        best_t, best_cost = 0.5, -np.inf
        for t in np.arange(0.01, 0.99, 0.01):
            y_pred = (y_prob >= t).astype(int)
            if y_pred.sum() == 0:
                continue
            cost, *_ = self.business_cost(y_true, y_pred)
            if cost > best_cost:
                best_cost = cost
                best_t = t
        return best_t

    def evaluate(self, name: str, y_true, y_prob) -> Dict:
        threshold = self.optimal_threshold(y_true, y_prob)
        y_pred = (y_prob >= threshold).astype(int)
        cost, tn, fp, fn, tp = self.business_cost(y_true, y_pred)
        result = {
            "name": name,
            "threshold": round(float(threshold), 4),
            "auc_roc": round(float(roc_auc_score(y_true, y_prob)), 6),
            "avg_precision": round(float(average_precision_score(y_true, y_prob)), 6),
            "f1": round(float(f1_score(y_true, y_pred)), 6),
            "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 6),
            "recall": round(float(recall_score(y_true, y_pred)), 6),
            "mcc": round(float(matthews_corrcoef(y_true, y_pred)), 6),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "business_value": cost, "y_prob": y_prob
        }
        logger.info(f"{name} | AUC={result['auc_roc']} | F1={result['f1']} | Value=${cost:,}")
        return result

    def select_best(self, results: list) -> Dict:
        best = max(results, key=lambda x: x["auc_roc"])
        logger.info(f"Best: {best['name']} AUC={best['auc_roc']}")
        return best

    def save_report(self, best: Dict, feature_count: int, n_test: int, n_attacks: int):
        report = {
            "best_model": str(best["name"]),
            "auc_roc": float(best["auc_roc"]),
            "avg_precision": float(best["avg_precision"]),
            "f1": float(best["f1"]),
            "precision": float(best["precision"]),
            "recall": float(best["recall"]),
            "mcc": float(best["mcc"]),
            "optimal_threshold": float(best["threshold"]),
            "attacks_caught": int(best["tp"]),
            "attacks_missed": int(best["fn"]),
            "false_alarms": int(best["fp"]),
            "business_value": int(best["business_value"]),
            "n_test": int(n_test),
            "n_test_attacks": int(n_attacks),
            "feature_count": int(feature_count)
        }
        with open(MODEL_REPORT_PATH, "w") as f:
            json.dump(report, f, indent=4)
        logger.info(f"Report saved to {MODEL_REPORT_PATH}")
        return report