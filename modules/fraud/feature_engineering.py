import pandas as pd
import numpy as np
from pathlib import Path
from typing import List
import sys
sys.path.append(str(Path(__file__).resolve().parent))
from config import TOP_PCA_FEATURES, VELOCITY_WINDOWS
from logger import get_logger

logger = get_logger("feature_engineering")


class FraudFeatureEngineer:

    def engineer_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df["hour_of_day"] = (df["Time"] / 3600) % 24
        df["day_number"] = (df["Time"] / 3600 / 24).astype(int) + 1
        df["is_night"] = ((df["hour_of_day"] >= 22) | (df["hour_of_day"] <= 5)).astype(int)
        df["is_peak_fraud_hour"] = ((df["hour_of_day"] >= 1) & (df["hour_of_day"] <= 3)).astype(int)
        logger.info("Time features engineered")
        return df

    def engineer_amount_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df["amount_log"] = np.log1p(df["Amount"])
        df["is_zero_amount"] = (df["Amount"] == 0).astype(int)
        df["is_small_amount"] = (df["Amount"] < 10).astype(int)
        df["is_round_amount"] = (df["Amount"] % 1 == 0).astype(int)
        df["amount_above_99p"] = (df["Amount"] > df["Amount"].quantile(0.99)).astype(int)
        logger.info("Amount features engineered")
        return df

    def engineer_pca_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        df["V14_V3_interaction"] = df["V14"] * df["V3"]
        df["V17_V12_interaction"] = df["V17"] * df["V12"]
        df["V14_V17_euclidean"] = np.sqrt(df["V14"] ** 2 + df["V17"] ** 2)
        df["top_features_sum_abs"] = df[TOP_PCA_FEATURES].abs().sum(axis=1)
        df["top_features_max_abs"] = df[TOP_PCA_FEATURES].abs().max(axis=1)
        for feat in TOP_PCA_FEATURES:
            df[f"{feat}_abs"] = df[feat].abs()
        logger.info("PCA interaction features engineered")
        return df

    def engineer_velocity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_sorted = df.sort_values("Time").reset_index(drop=True)
        for window in VELOCITY_WINDOWS:
            counts, amounts = [], []
            for _, row in df_sorted.iterrows():
                window_df = df_sorted[
                    (df_sorted["Time"] >= row["Time"] - window) &
                    (df_sorted["Time"] < row["Time"])
                ]
                counts.append(len(window_df))
                amounts.append(window_df["Amount"].sum())
            df_sorted[f"txn_count_{window}s"] = counts
            df_sorted[f"amount_sum_{window}s"] = amounts

        velocity_cols = (
            [f"txn_count_{w}s" for w in VELOCITY_WINDOWS] +
            [f"amount_sum_{w}s" for w in VELOCITY_WINDOWS]
        )
        df = df.merge(
            df_sorted[["Time", "Amount"] + velocity_cols].drop_duplicates(),
            on=["Time", "Amount"], how="left"
        )
        logger.info("Velocity features engineered")
        return df

    def drop_raw_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        drop_cols = ["Time", "Amount", "time_bin", "amount_quartile"]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])
        df = df.fillna(0)
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Starting feature engineering on {len(df):,} rows")
        df = self.engineer_time_features(df)
        df = self.engineer_amount_features(df)
        df = self.engineer_pca_interactions(df)
        df = self.engineer_velocity_features(df)
        df = self.drop_raw_columns(df)
        logger.info(f"Feature engineering complete — {df.shape[1]} total columns")
        return df