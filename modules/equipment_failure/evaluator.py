import numpy as np
import json
from typing import Dict, Tuple
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent))
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, confusion_matrix
from config import FAILURE_ALERT_THRESHOLD, COST_FALSE_ALARM, COST_MISSED_FAILURE, REWARD_CAUGHT_FAILURE, MODEL_REPORT_PATH
from logger import get_logger

logger = get_logger("evaluator")


class EquipmentEvaluator:

    def business_cost(self, y_true, y_pred) -> Tuple[int, int, int, int, int]:
        y_pred_class = (y_pred <= FAILURE_ALERT_THRESHOLD).astype(int)
        y_true_class = (np.array(y_true) <= FAILURE_ALERT_THRESHOLD).astype(int)
        if y_pred_class.sum() > 0 and y_true_class.sum() > 0:
            tn, fp, fn, tp = confusion_matrix(y_true_class, y_pred_class).ravel()
        else:
            tp = fp = fn = tn = 0
        cost = int(tp) * REWARD_CAUGHT_FAILURE + int(fn) * -COST_MISSED_FAILURE + int(fp) * -COST_FALSE_ALARM
        return int(cost), int(tn), int(fp), int(fn), int(tp)

    def nasa_score(self, y_true, y_pred) -> float:
        score = 0.0
        for true, pred in zip(y_true, y_pred):
            diff = float(pred) - float(true)
            score += (np.exp(-diff / 13) - 1) if diff < 0 else (np.exp(diff / 10) - 1)
        return float(score)

    def evaluate(self, name: str, y_true, y_pred) -> Dict:
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))
        nasa = self.nasa_score(y_true, y_pred)
        cost, tn, fp, fn, tp = self.business_cost(y_true, y_pred)
        result = {
            "name": name, "rmse": rmse, "mae": mae, "r2": r2,
            "nasa_score": nasa, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "business_value": cost, "y_pred": y_pred
        }
        logger.info(f"{name} | RMSE={rmse:.4f} | MAE={mae:.4f} | R2={r2:.4f} | Value=${cost:,}")
        return result

    def select_best(self, results: list) -> Dict:
        best = min(results, key=lambda x: x["rmse"])
        logger.info(f"Best model: {best['name']} RMSE={best['rmse']:.4f}")
        return best

    def save_report(self, best: Dict, feature_count: int, n_test: int):
        report = {
            "best_model": str(best["name"]),
            "rmse": float(best["rmse"]),
            "mae": float(best["mae"]),
            "r2": float(best["r2"]),
            "nasa_score": float(best["nasa_score"]),
            "failures_caught": int(best["tp"]),
            "failures_missed": int(best["fn"]),
            "false_alarms": int(best["fp"]),
            "business_value": int(best["business_value"]),
            "alert_threshold": int(FAILURE_ALERT_THRESHOLD),
            "n_test_engines": int(n_test),
            "feature_count": int(feature_count)
        }
        with open(MODEL_REPORT_PATH, "w") as f:
            json.dump(report, f, indent=4)
        logger.info(f"Report saved to {MODEL_REPORT_PATH}")
        return report