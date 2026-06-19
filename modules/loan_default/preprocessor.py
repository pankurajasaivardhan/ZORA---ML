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
    RANDOM_STATE, SCALER_PATH, FEATURE_NAMES_PATH, TARGET_COLUMN
)
from logger import get_logger

logger = get_logger("preprocessor")


class LoanPreprocessor:
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
        logger.info(f"Train: {len(df_train):,} | Test: {len(df_test):,}")
        logger.info(f"Train defaults: {df_train[TARGET_COLUMN].sum():,} | Test defaults: {df_test[TARGET_COLUMN].sum():,}")
        return df_train, df_test

    def fit_transform(self, df_train: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        self.feature_cols = [c for c in df_train.columns if c != TARGET_COLUMN]
        X = df_train[self.feature_cols]
        y = df_train[TARGET_COLUMN]
        logger.info("Fitting RobustScaler")
        X_scaled = pd.DataFrame(self.scaler.fit_transform(X), columns=self.feature_cols)
        logger.info("Applying SMOTE")
        X_res, y_res = self.smote.fit_resample(X_scaled, y)
        logger.info(f"After SMOTE — Non-default: {(y_res==0).sum():,} | Default: {(y_res==1).sum():,}")
        self.is_fitted = True
        joblib.dump(self.scaler, SCALER_PATH)
        joblib.dump(self.feature_cols, FEATURE_NAMES_PATH)
        logger.info("Scaler and feature names saved")
        return X_res, y_res

    def transform(self, df_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        if not self.is_fitted:
            raise RuntimeError("Preprocessor not fitted")
        X = df_test[self.feature_cols]
        y = df_test[TARGET_COLUMN]
        X_scaled = pd.DataFrame(self.scaler.transform(X), columns=self.feature_cols)
        return X_scaled, y

    def transform_single(self, record: dict) -> pd.DataFrame:
        if not self.is_fitted:
            self.scaler = joblib.load(SCALER_PATH)
            self.feature_cols = joblib.load(FEATURE_NAMES_PATH)
            self.is_fitted = True
        df = pd.DataFrame([record])[self.feature_cols]
        return pd.DataFrame(self.scaler.transform(df), columns=self.feature_cols)