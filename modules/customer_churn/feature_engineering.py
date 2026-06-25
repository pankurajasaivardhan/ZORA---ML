import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent))
from config import (
    BINARY_MAP, INTERNET_MAP, CONTRACT_MAP, PAYMENT_MAP,
    SERVICE_MAP, SERVICE_COLS
)
from logger import get_logger

logger = get_logger("feature_engineering")


class ChurnFeatureEngineer:

    def encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        binary_cols = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]
        for col in binary_cols:
            if col in df.columns:
                df[col + "_encoded"] = df[col].map(BINARY_MAP).fillna(0)

        df["InternetService_encoded"] = df["InternetService"].map(INTERNET_MAP).fillna(0)
        df["Contract_encoded"] = df["Contract"].map(CONTRACT_MAP).fillna(0)
        df["PaymentMethod_encoded"] = df["PaymentMethod"].map(PAYMENT_MAP).fillna(0)

        for col in SERVICE_COLS:
            df[col + "_encoded"] = df[col].map(SERVICE_MAP).fillna(0)

        df["churn_risk_contract"] = (df["Contract_encoded"] == 0).astype(int)
        df["churn_risk_internet"] = (df["InternetService_encoded"] == 2).astype(int)
        return df

    def financial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["charges_per_month_tenure"] = df["MonthlyCharges"] / (df["tenure"] + 1)
        df["total_charges_expected"] = df["MonthlyCharges"] * df["tenure"]
        df["charges_deviation"] = df["TotalCharges"] - df["total_charges_expected"]
        df["charges_deviation_pct"] = df["charges_deviation"] / (df["total_charges_expected"] + 1)
        df["high_monthly_charges"] = (df["MonthlyCharges"] > df["MonthlyCharges"].quantile(0.75)).astype(int)
        df["low_tenure"] = (df["tenure"] <= 12).astype(int)
        df["long_tenure"] = (df["tenure"] >= 48).astype(int)
        df["revenue_tier"] = pd.qcut(df["MonthlyCharges"], q=4, labels=[0,1,2,3], duplicates="drop").astype(int)

        service_encoded = [c for c in df.columns if c.endswith("_encoded") and
                           any(s in c for s in SERVICE_COLS)]
        df["total_services"] = df[service_encoded].clip(lower=0).sum(axis=1)
        df["no_security_services"] = (
            (df.get("OnlineSecurity_encoded", 0) == 0) &
            (df.get("DeviceProtection_encoded", 0) == 0) &
            (df.get("OnlineBackup_encoded", 0) == 0)
        ).astype(int)

        df["customer_lifetime_value"] = df["TotalCharges"]
        df["monthly_commitment_score"] = df["Contract_encoded"] * 2 + df["tenure"] / 12
        df["churn_risk_score"] = (
            (1 - df["Contract_encoded"] / 2) * 0.35 +
            (df["InternetService_encoded"] == 2).astype(float) * 0.20 +
            df["low_tenure"].astype(float) * 0.25 +
            df["high_monthly_charges"].astype(float) * 0.10 +
            (1 - df.get("Partner_encoded", 0)) * 0.10
        )
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Engineering features on {len(df):,} rows")
        df = self.encode_categoricals(df)
        df = self.financial_features(df)
        df = df.fillna(0)
        logger.info(f"Feature engineering complete: {df.shape[1]} columns")
        return df 