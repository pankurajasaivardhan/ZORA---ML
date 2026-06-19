import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent))
from config import HEART_ENGINEERED_PATH, DIABETES_ENGINEERED_PATH
from logger import get_logger

logger = get_logger("feature_engineering")


class HealthFeatureEngineer:

    def engineer_heart(self, df: pd.DataFrame) -> pd.DataFrame:
        h = df.copy()
        h["age_encoded"] = pd.cut(h["age"], bins=[0,40,50,60,70,100],
                                   labels=[1,2,3,4,5]).astype(float)
        h["high_bp"] = (h["trestbps"] > 140).astype(int)
        h["high_chol"] = (h["chol"] > 240).astype(int)
        h["low_max_hr"] = (h["thalach"] < 120).astype(int)
        h["hr_reserve"] = 220 - h["age"] - h["thalach"]
        h["hr_reserve_pct"] = h["hr_reserve"] / (220 - h["age"])
        h["chest_pain_severity"] = h["cp"].map({0: 0, 1: 3, 2: 2, 3: 1})
        h["cardiac_risk_score"] = (
            h["age_encoded"].fillna(3) / 5 * 0.20 +
            (h["trestbps"] / 200) * 0.15 +
            (h["chol"] / 600) * 0.10 +
            h["exang"] * 0.20 +
            (h["oldpeak"] / 6.2) * 0.20 +
            (h["ca"] / 4) * 0.15
        )
        h["has_exercise_angina"] = h["exang"]
        h["multiple_risk_factors"] = (
            (h["high_bp"] + h["high_chol"] + h["fbs"] + h["exang"]) >= 2
        ).astype(int)
        h["age_encoded"] = h["age_encoded"].fillna(3)
        h = h.fillna(0)
        logger.info(f"Heart features engineered: {h.shape[1]} columns")
        return h

    def engineer_diabetes(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        d["high_glucose"] = (d["Glucose"] > 140).astype(int)
        d["obese"] = (d["BMI"] >= 30).astype(int)
        d["high_age"] = (d["Age"] >= 45).astype(int)
        d["high_pregnancies"] = (d["Pregnancies"] >= 4).astype(int)
        d["glucose_bmi_interaction"] = d["Glucose"] * d["BMI"] / 1000
        d["insulin_resistance"] = d["Glucose"] / (d["Insulin"] + 1)
        d["bmi_age_risk"] = d["BMI"] * d["Age"] / 1000
        d["glucose_normalized"] = (d["Glucose"] - d["Glucose"].min()) / (d["Glucose"].max() - d["Glucose"].min() + 1e-10)
        d["bmi_normalized"] = (d["BMI"] - d["BMI"].min()) / (d["BMI"].max() - d["BMI"].min() + 1e-10)
        d["age_normalized"] = (d["Age"] - d["Age"].min()) / (d["Age"].max() - d["Age"].min() + 1e-10)
        d["diabetes_risk_score"] = (
            d["glucose_normalized"] * 0.40 +
            d["bmi_normalized"] * 0.25 +
            d["age_normalized"] * 0.15 +
            (d["Pregnancies"] / (d["Pregnancies"].max() + 1e-10)) * 0.10 +
            d["DiabetesPedigreeFunction"] / (d["DiabetesPedigreeFunction"].max() + 1e-10) * 0.10
        )
        d["multiple_risk_factors"] = (
            (d["high_glucose"] + d["obese"] + d["high_age"] + d["high_pregnancies"]) >= 2
        ).astype(int)
        d = d.fillna(0)
        logger.info(f"Diabetes features engineered: {d.shape[1]} columns")
        return d