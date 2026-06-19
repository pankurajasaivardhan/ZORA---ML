import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, Tuple
import sys
sys.path.append(str(Path(__file__).resolve().parent))
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, matthews_corrcoef, confusion_matrix
)
from config import COST_FALSE_NEGATIVE, COST_FALSE_POSITIVE, REWARD_TRUE_POSITIVE, MODEL_REPORT_PATH
from logger import get_logger

logger = get_logger("evaluator")


class LoanEvaluator:

    def business_cost(self, y_true, y_pred) -> Tuple[int, int, int, int, int]:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        cost = (tp * REWARD_TRUE_POSITIVE) + (fn * -COST_FALSE_NEGATIVE) + (fp * -COST_FALSE_POSITIVE)
        return cost, tn, fp, fn, tp

    def optimal_threshold(self, y_true, y_prob) -> Tuple[float, int]:
        best_t, best_cost = 0.5, -np.inf
        for t in np.arange(0.01, 0.99, 0.005):
            y_pred = (y_prob >= t).astype(int)
            if y_pred.sum() == 0:
                continue
            cost, *_ = self.business_cost(y_true, y_pred)
            if cost > best_cost:
                best_cost = cost
                best_t = t
        return best_t, best_cost

    def evaluate(self, name: str, y_true, y_prob) -> Dict:
        threshold, _ = self.optimal_threshold(y_true, y_prob)
        y_pred = (y_prob >= threshold).astype(int)
        cost, tn, fp, fn, tp = self.business_cost(y_true, y_pred)
        result = {
            "name": name,
            "threshold": round(float(threshold), 4),
            "auc_roc": round(float(roc_auc_score(y_true, y_prob)), 6),
            "avg_precision": round(float(average_precision_score(y_true, y_prob)), 6),
            "f1": round(float(f1_score(y_true, y_pred)), 6),
            "precision": round(float(precision_score(y_true, y_pred)), 6),
            "recall": round(float(recall_score(y_true, y_pred)), 6),
            "mcc": round(float(matthews_corrcoef(y_true, y_pred)), 6),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            "business_value": int(cost),
            "y_prob": y_prob
        }
        logger.info(
            f"{name} | AUC={result['auc_roc']} | F1={result['f1']} | "
            f"Recall={result['recall']} | Caught={tp} | Missed={fn} | Value=${cost:,}"
        )
        return result

    def select_best(self, results: list) -> Dict:
        best = max(results, key=lambda x: x["auc_roc"])
        logger.info(f"Best model: {best['name']} AUC={best['auc_roc']}")
        return best

    def save_report(self, best: Dict, feature_count: int, test_size: int, test_defaults: int):
        report = {
            "best_model": best["name"],
            "optimal_threshold": best["threshold"],
            "auc_roc": best["auc_roc"],
            "avg_precision": best["avg_precision"],
            "f1": best["f1"],
            "precision": best["precision"],
            "recall": best["recall"],
            "mcc": best["mcc"],
            "defaults_caught": best["tp"],
            "defaults_missed": best["fn"],
            "false_alarms": best["fp"],
            "business_value": best["business_value"],
            "test_size": test_size,
            "test_defaults": test_defaults,
            "feature_count": feature_count
        }
        with open(MODEL_REPORT_PATH, "w") as f:
            json.dump(report, f, indent=4)
        logger.info(f"Report saved to {MODEL_REPORT_PATH}")
        return report