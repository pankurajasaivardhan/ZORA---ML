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
        f"health_local_{filename.replace('.py', '')}",
        BASE_DIR / filename
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_config = _load_local("config.py")
_logger_mod = _load_local("logger.py")

HEART_MODEL_PATH = _config.HEART_MODEL_PATH
DIABETES_MODEL_PATH = _config.DIABETES_MODEL_PATH
SCALER_HEART_PATH = _config.SCALER_HEART_PATH
SCALER_DIABETES_PATH = _config.SCALER_DIABETES_PATH
HEART_FEATURES_PATH = _config.HEART_FEATURES_PATH
DIABETES_FEATURES_PATH = _config.DIABETES_FEATURES_PATH
MODEL_REPORT_PATH = _config.MODEL_REPORT_PATH
HEART_WEIGHT = _config.HEART_WEIGHT
DIABETES_WEIGHT = _config.DIABETES_WEIGHT

logger = _logger_mod.get_logger("health_predictor")


class HealthRiskPredictor:
    def __init__(self):
        self.heart_model = None
        self.diabetes_model = None
        self.scaler_heart = None
        self.scaler_diabetes = None
        self.heart_features = None
        self.diabetes_features = None
        self.heart_threshold = 0.5
        self.diabetes_threshold = 0.5
        self._loaded = False

    def load(self):
        self.heart_model = joblib.load(HEART_MODEL_PATH)
        self.diabetes_model = joblib.load(DIABETES_MODEL_PATH)
        self.scaler_heart = joblib.load(SCALER_HEART_PATH)
        self.scaler_diabetes = joblib.load(SCALER_DIABETES_PATH)
        self.heart_features = joblib.load(HEART_FEATURES_PATH)
        self.diabetes_features = joblib.load(DIABETES_FEATURES_PATH)
        with open(MODEL_REPORT_PATH) as f:
            report = json.load(f)
        self.heart_threshold = report.get("heart_threshold", 0.5)
        self.diabetes_threshold = report.get("diabetes_threshold", 0.5)
        self._loaded = True
        logger.info(f"Health predictor loaded | heart_threshold={self.heart_threshold} | diabetes_threshold={self.diabetes_threshold}")
        return self

    def _engineer_heart(self, record: dict) -> dict:
        r = record.copy()
        age = r.get("age", 55)
        trestbps = r.get("trestbps", 130)
        chol = r.get("chol", 240)
        thalach = r.get("thalach", 150)
        exang = r.get("exang", 0)
        oldpeak = r.get("oldpeak", 1.0)
        ca = r.get("ca", 0)
        if age < 40: r["age_encoded"] = 1
        elif age < 50: r["age_encoded"] = 2
        elif age < 60: r["age_encoded"] = 3
        elif age < 70: r["age_encoded"] = 4
        else: r["age_encoded"] = 5
        r["high_bp"] = int(trestbps > 140)
        r["high_chol"] = int(chol > 240)
        r["low_max_hr"] = int(thalach < 120)
        r["hr_reserve"] = 220 - age - thalach
        r["hr_reserve_pct"] = r["hr_reserve"] / (220 - age + 1e-10)
        r["chest_pain_severity"] = {0: 0, 1: 3, 2: 2, 3: 1}.get(r.get("cp", 0), 0)
        r["cardiac_risk_score"] = (
            r["age_encoded"] / 5 * 0.20 +
            (trestbps / 200) * 0.15 +
            (chol / 600) * 0.10 +
            exang * 0.20 +
            (oldpeak / 6.2) * 0.20 +
            (ca / 4) * 0.15
        )
        r["has_exercise_angina"] = exang
        r["multiple_risk_factors"] = int(
            (r["high_bp"] + r["high_chol"] + r.get("fbs", 0) + exang) >= 2
        )
        return r

    def _engineer_diabetes(self, record: dict) -> dict:
        r = record.copy()
        glucose = r.get("Glucose", 120)
        bmi = r.get("BMI", 28)
        age = r.get("Age", 35)
        insulin = r.get("Insulin", 80)
        pregnancies = r.get("Pregnancies", 2)
        dpf = r.get("DiabetesPedigreeFunction", 0.5)
        r["high_glucose"] = int(glucose > 140)
        r["obese"] = int(bmi >= 30)
        r["high_age"] = int(age >= 45)
        r["high_pregnancies"] = int(pregnancies >= 4)
        r["glucose_bmi_interaction"] = glucose * bmi / 1000
        r["insulin_resistance"] = glucose / (insulin + 1)
        r["bmi_age_risk"] = bmi * age / 1000
        r["glucose_normalized"] = (glucose - 44) / (199 - 44 + 1e-10)
        r["bmi_normalized"] = (bmi - 18) / (67 - 18 + 1e-10)
        r["age_normalized"] = (age - 21) / (81 - 21 + 1e-10)
        r["diabetes_risk_score"] = (
            r["glucose_normalized"] * 0.40 +
            r["bmi_normalized"] * 0.25 +
            r["age_normalized"] * 0.15 +
            (pregnancies / 17) * 0.10 +
            (dpf / 2.42) * 0.10
        )
        r["multiple_risk_factors"] = int(
            (r["high_glucose"] + r["obese"] + r["high_age"] + r["high_pregnancies"]) >= 2
        )
        return r

    def _scale_and_predict(self, record, features, scaler, model):
        df = pd.DataFrame([record])
        for col in features:
            if col not in df.columns:
                df[col] = 0
        df = df[features].fillna(0)
        df_scaled = pd.DataFrame(scaler.transform(df), columns=features)
        if hasattr(model, "predict_proba"):
            return float(model.predict_proba(df_scaled)[:, 1][0])
        return float(model.decision_function(df_scaled)[0])

    def predict(self, heart_record: dict, diabetes_record: dict) -> Dict:
        if not self._loaded:
            self.load()

        heart_engineered = self._engineer_heart(heart_record)
        diabetes_engineered = self._engineer_diabetes(diabetes_record)

        heart_score = self._scale_and_predict(
            heart_engineered, self.heart_features, self.scaler_heart, self.heart_model)
        diabetes_score = self._scale_and_predict(
            diabetes_engineered, self.diabetes_features, self.scaler_diabetes, self.diabetes_model)

        combined_score = HEART_WEIGHT * heart_score + DIABETES_WEIGHT * diabetes_score

        heart_risk = (
            "CRITICAL" if heart_score >= 0.80 else
            "HIGH" if heart_score >= 0.60 else
            "MEDIUM" if heart_score >= 0.40 else "LOW"
        )
        diabetes_risk = (
            "CRITICAL" if diabetes_score >= 0.80 else
            "HIGH" if diabetes_score >= 0.60 else
            "MEDIUM" if diabetes_score >= 0.40 else "LOW"
        )
        overall_risk = (
            "CRITICAL" if combined_score >= 0.75 else
            "HIGH" if combined_score >= 0.55 else
            "MEDIUM" if combined_score >= 0.35 else "LOW"
        )

        result = {
            "heart_disease_score": round(heart_score, 6),
            "diabetes_score": round(diabetes_score, 6),
            "combined_health_score": round(combined_score, 6),
            "heart_disease_detected": bool(heart_score >= self.heart_threshold),
            "diabetes_detected": bool(diabetes_score >= self.diabetes_threshold),
            "heart_risk_level": heart_risk,
            "diabetes_risk_level": diabetes_risk,
            "overall_risk_level": overall_risk,
            "recommendation": (
                "IMMEDIATE MEDICAL ATTENTION REQUIRED" if overall_risk == "CRITICAL" else
                "CONSULT DOCTOR SOON" if overall_risk == "HIGH" else
                "MONITOR REGULARLY" if overall_risk == "MEDIUM" else
                "MAINTAIN HEALTHY LIFESTYLE"
            )
        }
        logger.info(
            f"Health prediction | heart={heart_score:.4f} | diabetes={diabetes_score:.4f} | "
            f"combined={combined_score:.4f} | risk={overall_risk}"
        )
        return result