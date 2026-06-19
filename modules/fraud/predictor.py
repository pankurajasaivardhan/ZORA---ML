import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict
import sys
sys.path.append(str(Path(__file__).resolve().parent))
from config import BEST_MODEL_PATH, SCALER_PATH, FEATURE_NAMES_PATH, MODEL_REPORT_PATH
from logger import get_logger
import json

logger = get_logger("predictor")


class FraudPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.threshold = 0.5
        self._loaded = False

    def load(self):
        self.model = joblib.load(BEST_MODEL_PATH)
        self.scaler = joblib.load(SCALER_PATH)
        self.feature_names = joblib.load(FEATURE_NAMES_PATH)
        with open(MODEL_REPORT_PATH) as f:
            report = json.load(f)
        self.threshold = report.get("optimal_threshold", 0.5)
        self._loaded = True
        logger.info(f"Predictor loaded | threshold={self.threshold}")
        return self

    def _validate_input(self, record: Dict):
        missing = [f for f in self.feature_names if f not in record]
        if missing:
            raise ValueError(f"Missing features: {missing}")

    def predict(self, record: Dict) -> Dict:
        if not self._loaded:
            self.load()
        self._validate_input(record)
        df = pd.DataFrame([record])[self.feature_names]
        df_scaled = pd.DataFrame(
            self.scaler.transform(df),
            columns=self.feature_names
        )
        if hasattr(self.model, "predict_proba"):
            fraud_score = float(self.model.predict_proba(df_scaled)[:, 1][0])
        else:
            raw = self.model.decision_function(df_scaled)
            fraud_score = float(1 - (raw[0] - raw.min()) / (raw.max() - raw.min() + 1e-10))

        is_fraud = fraud_score >= self.threshold
        risk_level = (
            "CRITICAL" if fraud_score >= 0.85 else
            "HIGH" if fraud_score >= 0.65 else
            "MEDIUM" if fraud_score >= 0.40 else
            "LOW"
        )
        result = {
            "fraud_score": round(fraud_score, 6),
            "is_fraud": bool(is_fraud),
            "risk_level": risk_level,
            "threshold_used": self.threshold,
            "action": "BLOCK" if is_fraud else "APPROVE"
        }
        logger.info(
            f"Prediction | score={fraud_score:.4f} | fraud={is_fraud} | "
            f"risk={risk_level} | action={result['action']}"
        )
        return result