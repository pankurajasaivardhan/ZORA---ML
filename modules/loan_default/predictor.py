import json
import importlib.util
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict

BASE_DIR = Path(__file__).resolve().parent


def _load_local(filename: str):
    spec = importlib.util.spec_from_file_location(
        f"loan_local_{filename.replace('.py', '')}",
        BASE_DIR / filename
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_config = _load_local("config.py")
_logger_mod = _load_local("logger.py")

BEST_MODEL_PATH = _config.BEST_MODEL_PATH
SCALER_PATH = _config.SCALER_PATH
FEATURE_NAMES_PATH = _config.FEATURE_NAMES_PATH
MODEL_REPORT_PATH = _config.MODEL_REPORT_PATH
GRADE_MAP = _config.GRADE_MAP
HOME_MAP = _config.HOME_MAP

logger = _logger_mod.get_logger("loan_predictor")


class LoanPredictor:
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
        logger.info(f"Loan predictor loaded | threshold={self.threshold}")
        return self

    def _engineer_features(self, record: dict) -> dict:
        r = record.copy()
        r["grade_encoded"] = GRADE_MAP.get(r.get("grade", "C"), 3)
        r["home_encoded"] = HOME_MAP.get(r.get("home_ownership", "RENT"), 1)
        r["purpose_risk_score"] = 0.15
        r["loan_to_income_ratio"] = r.get("loan_amnt", 0) / (r.get("annual_inc", 1) + 1)
        r["installment_to_income"] = r.get("installment", 0) / (r.get("annual_inc", 1) / 12 + 1)
        r["revolving_burden"] = r.get("revol_bal", 0) / (r.get("annual_inc", 1) + 1)
        r["credit_utilization"] = r.get("revol_util", 0) / 100
        r["fico_avg"] = (r.get("fico_range_low", 700) + r.get("fico_range_high", 720)) / 2
        r["fico_normalized"] = (r["fico_avg"] - 580) / (850 - 580 + 1e-10)
        r["int_rate_normalized"] = (r.get("int_rate", 12) - 5) / (30 - 5 + 1e-10)
        r["composite_risk_score"] = (
            r["int_rate_normalized"] * 0.35 +
            r["grade_encoded"] / 7 * 0.30 +
            (1 - r["fico_normalized"]) * 0.25 +
            min(r.get("dti", 15) / 40, 1) * 0.10
        )
        r["high_dti"] = int(r.get("dti", 0) > 30)
        r["high_int_rate"] = int(r.get("int_rate", 0) > 20)
        r["low_fico"] = int(r.get("fico_range_low", 700) < 660)
        r["risky_grade"] = int(r["grade_encoded"] >= 5)
        r["delinq_per_year"] = r.get("delinq_2yrs", 0) / 2
        r["has_delinquency"] = int(r.get("delinq_2yrs", 0) > 0)
        r["has_public_record"] = int(r.get("pub_rec", 0) > 0)
        r["has_bankruptcy"] = int(r.get("pub_rec_bankruptcies", 0) > 0)
        r["credit_breadth"] = r.get("open_acc", 5) / (r.get("total_acc", 10) + 1)
        r["mort_acc_flag"] = int(r.get("mort_acc", 0) > 0)
        emp = r.get("emp_length", 5)
        r["emp_length_clean"] = emp if emp is not None else 5
        r["experienced_borrower"] = int(r["emp_length_clean"] >= 5)
        r["multiple_derogatory"] = int(r.get("delinq_2yrs", 0) > 1 or r.get("pub_rec", 0) > 1)
        return r

    def predict(self, record: dict) -> Dict:
        if not self._loaded:
            self.load()
        engineered = self._engineer_features(record)
        df = pd.DataFrame([engineered])
        missing = [f for f in self.feature_names if f not in df.columns]
        for col in missing:
            df[col] = 0
        df = df[self.feature_names]
        df_scaled = pd.DataFrame(self.scaler.transform(df), columns=self.feature_names)
        if hasattr(self.model, "predict_proba"):
            default_score = float(self.model.predict_proba(df_scaled)[:, 1][0])
        else:
            default_score = float(self.model.decision_function(df_scaled)[0])
        is_default = default_score >= self.threshold
        risk_level = (
            "CRITICAL" if default_score >= 0.75 else
            "HIGH" if default_score >= 0.50 else
            "MEDIUM" if default_score >= 0.30 else
            "LOW"
        )
        result = {
            "default_score": round(default_score, 6),
            "is_default": bool(is_default),
            "risk_level": risk_level,
            "threshold_used": self.threshold,
            "decision": "REJECT" if is_default else "APPROVE",
            "composite_risk_score": round(float(engineered["composite_risk_score"]), 4),
            "grade_encoded": int(engineered["grade_encoded"])
        }
        logger.info(
            f"Loan prediction | score={default_score:.4f} | decision={result['decision']} | risk={risk_level}"
        )
        return result