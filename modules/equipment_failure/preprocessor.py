import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Tuple, List
import sys
sys.path.append(str(Path(__file__).resolve().parent))
from sklearn.preprocessing import RobustScaler
from config import SCALER_PATH, FEATURE_NAMES_PATH, CLIP_RUL, FAILURE_ALERT_THRESHOLD
from logger import get_logger

logger = get_logger("preprocessor")


class EquipmentPreprocessor:
    def __init__(self):
        self.scaler = RobustScaler()
        self.feature_cols = None
        self.is_fitted = False

    def prepare(self, train_df: pd.DataFrame, test_df: pd.DataFrame, meta: dict):
        drop_cols = ["unit_id", "RUL", "RUL_clipped"] + [f"failure_within_{t}" for t in [15, 30, 50]]
        self.feature_cols = [c for c in meta["feature_cols"] if c in train_df.columns and c not in drop_cols]

        X_train = train_df[self.feature_cols].fillna(0)
        y_train = train_df["RUL_clipped"]
        X_test = test_df[self.feature_cols].fillna(0)
        y_test = test_df["RUL_true"]

        logger.info("Fitting RobustScaler")
        X_train_s = pd.DataFrame(self.scaler.fit_transform(X_train), columns=self.feature_cols)
        X_test_s = pd.DataFrame(self.scaler.transform(X_test), columns=self.feature_cols)

        joblib.dump(self.scaler, SCALER_PATH)
        joblib.dump(self.feature_cols, FEATURE_NAMES_PATH)
        self.is_fitted = True
        logger.info(f"Features: {len(self.feature_cols)} | Train: {len(X_train)} | Test: {len(X_test)}")
        return X_train_s, X_test_s, y_train, y_test

    def transform_single(self, record: dict) -> pd.DataFrame:
        if not self.is_fitted:
            self.scaler = joblib.load(SCALER_PATH)
            self.feature_cols = joblib.load(FEATURE_NAMES_PATH)
            self.is_fitted = True
        df = pd.DataFrame([record])
        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = 0
        df = df[self.feature_cols].fillna(0)
        return pd.DataFrame(self.scaler.transform(df), columns=self.feature_cols)