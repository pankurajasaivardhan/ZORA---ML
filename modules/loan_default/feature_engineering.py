import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent))
from config import GRADE_MAP, HOME_MAP, ENGINEERED_DATA_PATH
from logger import get_logger

logger = get_logger("feature_engineering")


class LoanFeatureEngineer:

    def encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        df["grade_encoded"] = df["grade"].map(GRADE_MAP)
        df["home_encoded"] = df["home_ownership"].map(HOME_MAP).fillna(1)
        purpose_rates = df.groupby("purpose")["is_default"].mean()
        df["purpose_risk_score"] = df["purpose"].map(purpose_rates)
        return df

    def financial_risk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df["loan_to_income_ratio"] = df["loan_amnt"] / (df["annual_inc"] + 1)
        df["installment_to_income"] = df["installment"] / (df["annual_inc"] / 12 + 1)
        df["revolving_burden"] = df["revol_bal"] / (df["annual_inc"] + 1)
        df["credit_utilization"] = df["revol_util"] / 100
        df["fico_avg"] = (df["fico_range_low"] + df["fico_range_high"]) / 2
        df["fico_normalized"] = (df["fico_avg"] - df["fico_avg"].min()) / (df["fico_avg"].max() - df["fico_avg"].min() + 1e-10)
        df["int_rate_normalized"] = (df["int_rate"] - df["int_rate"].min()) / (df["int_rate"].max() - df["int_rate"].min() + 1e-10)
        df["composite_risk_score"] = (
            df["int_rate_normalized"] * 0.35 +
            df["grade_encoded"] / 7 * 0.30 +
            (1 - df["fico_normalized"]) * 0.25 +
            (df["dti"] / df["dti"].quantile(0.99).clip(1)).clip(0, 1) * 0.10
        )
        df["high_dti"] = (df["dti"] > 30).astype(int)
        df["high_int_rate"] = (df["int_rate"] > 20).astype(int)
        df["low_fico"] = (df["fico_range_low"] < 660).astype(int)
        df["risky_grade"] = (df["grade_encoded"] >= 5).astype(int)
        return df

    def credit_history_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df["delinq_per_year"] = df["delinq_2yrs"] / 2
        df["has_delinquency"] = (df["delinq_2yrs"] > 0).astype(int)
        df["has_public_record"] = (df["pub_rec"] > 0).astype(int)
        df["has_bankruptcy"] = (df["pub_rec_bankruptcies"] > 0).astype(int)
        df["credit_breadth"] = df["open_acc"] / (df["total_acc"] + 1)
        df["mort_acc_flag"] = (df["mort_acc"] > 0).astype(int).fillna(0)
        df["emp_length_clean"] = df["emp_length"].fillna(df["emp_length"].median())
        df["experienced_borrower"] = (df["emp_length_clean"] >= 5).astype(int)
        df["multiple_derogatory"] = ((df["delinq_2yrs"] > 1) | (df["pub_rec"] > 1)).astype(int)
        return df

    def drop_raw_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        drop_cols = ["grade", "home_ownership", "purpose", "emp_length"]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])
        df = df.fillna(0)
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Starting feature engineering on {len(df):,} rows")
        df = self.encode_categoricals(df)
        df = self.financial_risk_features(df)
        df = self.credit_history_features(df)
        df = self.drop_raw_columns(df)
        logger.info(f"Feature engineering complete — {df.shape[1]} total columns")
        return df