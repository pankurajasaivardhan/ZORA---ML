import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Tuple
import sys
sys.path.append(str(Path(__file__).resolve().parent))
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE
from config import (
    TRAIN_RATIO, SMOTE_SAMPLING_STRATEGY, SMOTE_K_NEIGHBORS,
    RANDOM_STATE, SCALER_PATH, TARGET_COLUMN
)
from logger import get_logger

logger = get_logger("preprocessor")


class FraudPreprocessor:
    def __init__(self):
        self.scaler = RobustScaler()
        self.smote = SMOTE(
            sampling_strategy=SMOTE_SAMPLING_STRATEGY,
            random_state=RANDOM_STATE,
            k_neighbors=SMOTE_K_NEIGHBORS,
            n_jobs=-1
        )
        self.feature_cols = None
        self.is_fitted = False

    def temporal_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        split_idx = int(len(df) * TRAIN_RATIO)
        df_train = df.iloc[:split_idx].copy()
        df_test = df.iloc[split_idx:].copy()
        logger.info(f"Train: {len(df_train):,} rows | Test: {len(df_test):,} rows")
        logger.info(f"Train fraud: {df_train[TARGET_COLUMN].sum()} | Test fraud: {df_test[TARGET_COLUMN].sum()}")
        return df_train, df_test

    def fit_transform(
        self, df_train: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        self.feature_cols = [c for c in df_train.columns if c != TARGET_COLUMN]
        X_train = df_train[self.feature_cols]
        y_train = df_train[TARGET_COLUMN]

        logger.info("Fitting RobustScaler on training data")
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=self.feature_cols
        )

        logger.info("Applying SMOTE oversampling on training data only")
        X_resampled, y_resampled = self.smote.fit_resample(X_scaled, y_train)
        logger.info(f"After SMOTE — Legitimate: {(y_resampled==0).sum():,} | Fraud: {(y_resampled==1).sum():,}")

        self.is_fitted = True
        joblib.dump(self.scaler, SCALER_PATH)
        logger.info(f"Scaler saved to {SCALER_PATH}")
        return X_resampled, y_resampled

    def transform(self, df_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before calling transform")
        X_test = df_test[self.feature_cols]
        y_test = df_test[TARGET_COLUMN]
        X_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=self.feature_cols
        )
        return X_scaled, y_test

    def transform_single(self, record: dict) -> pd.DataFrame:
        if not self.is_fitted:
            self.scaler = joblib.load(SCALER_PATH)
            self.is_fitted = True
        df = pd.DataFrame([record])
        return pd.DataFrame(
            self.scaler.transform(df),
            columns=df.columns
        )