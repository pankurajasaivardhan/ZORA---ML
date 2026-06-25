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
        f"churn_local_{filename.replace('.py', '')}",
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
BINARY_MAP = _config.BINARY_MAP
INTERNET_MAP = _config.INTERNET_MAP
CONTRACT_MAP = _config.CONTRACT_MAP
PAYMENT_MAP = _config.PAYMENT_MAP
SERVICE_MAP = _config.SERVICE_MAP
SERVICE_COLS = _config.SERVICE_COLS

logger = _logger_mod.get_logger("churn_predictor")


class ChurnPredictor:
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
        logger.info(f"Churn predictor loaded | threshold={self.threshold}")
        return self

    def _engineer_features(self, record: dict) -> dict:
        r = record.copy()

        for col in ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]:
            r[col + "_encoded"] = BINARY_MAP.get(r.get(col, "No"), 0)

        r["InternetService_encoded"] = INTERNET_MAP.get(r.get("InternetService", "No"), 0)
        r["Contract_encoded"] = CONTRACT_MAP.get(r.get("Contract", "Month-to-month"), 0)
        r["PaymentMethod_encoded"] = PAYMENT_MAP.get(r.get("PaymentMethod", "Electronic check"), 0)

        for col in SERVICE_COLS:
            r[col + "_encoded"] = SERVICE_MAP.get(r.get(col, "No"), 0)

        r["churn_risk_contract"] = int(r["Contract_encoded"] == 0)
        r["churn_risk_internet"] = int(r["InternetService_encoded"] == 2)

        tenure = r.get("tenure", 12)
        monthly = r.get("MonthlyCharges", 65)
        total = r.get("TotalCharges", monthly * tenure)

        r["charges_per_month_tenure"] = monthly / (tenure + 1)
        r["total_charges_expected"] = monthly * tenure
        r["charges_deviation"] = total - r["total_charges_expected"]
        r["charges_deviation_pct"] = r["charges_deviation"] / (r["total_charges_expected"] + 1)
        r["high_monthly_charges"] = int(monthly > 80)
        r["low_tenure"] = int(tenure <= 12)
        r["long_tenure"] = int(tenure >= 48)
        r["revenue_tier"] = 0 if monthly < 35 else 1 if monthly < 70 else 2 if monthly < 90 else 3

        service_vals = [r.get(c + "_encoded", 0) for c in SERVICE_COLS]
        r["total_services"] = sum(max(v, 0) for v in service_vals)
        r["no_security_services"] = int(
            r.get("OnlineSecurity_encoded", 0) == 0 and
            r.get("DeviceProtection_encoded", 0) == 0 and
            r.get("OnlineBackup_encoded", 0) == 0
        )

        r["customer_lifetime_value"] = total
        r["monthly_commitment_score"] = r["Contract_encoded"] * 2 + tenure / 12
        r["churn_risk_score"] = (
            (1 - r["Contract_encoded"] / 2) * 0.35 +
            int(r["InternetService_encoded"] == 2) * 0.20 +
            r["low_tenure"] * 0.25 +
            r["high_monthly_charges"] * 0.10 +
            (1 - r["Partner_encoded"]) * 0.10
        )
        return r

    def predict(self, record: dict) -> Dict:
        if not self._loaded:
            self.load()

        engineered = self._engineer_features(record)
        df = pd.DataFrame([engineered])
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0
        df = df[self.feature_names].fillna(0)
        df_scaled = pd.DataFrame(self.scaler.transform(df), columns=self.feature_names)

        churn_score = float(self.model.predict_proba(df_scaled)[:, 1][0])
        will_churn = churn_score >= self.threshold

        risk_level = (
            "CRITICAL" if churn_score >= 0.75 else
            "HIGH" if churn_score >= 0.55 else
            "MEDIUM" if churn_score >= 0.35 else
            "LOW"
        )

        result = {
            "churn_score": round(churn_score, 6),
            "will_churn": bool(will_churn),
            "risk_level": risk_level,
            "threshold_used": self.threshold,
            "action": "RETAIN — Offer incentive" if will_churn else "MONITOR",
            "churn_risk_score": round(float(engineered["churn_risk_score"]), 4),
            "recommendation": (
                "URGENT: Contact immediately with retention offer" if risk_level == "CRITICAL" else
                "Proactive outreach recommended within 30 days" if risk_level == "HIGH" else
                "Monitor engagement, consider loyalty perks" if risk_level == "MEDIUM" else
                "Customer appears stable, no action needed"
            )
        }
        logger.info(f"Churn prediction | score={churn_score:.4f} | action={result['action']} | risk={risk_level}")
        return result