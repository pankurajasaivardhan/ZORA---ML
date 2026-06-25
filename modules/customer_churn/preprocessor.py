import pandas as pd
import joblib
from pathlib import Path
from typing import Tuple
import sys
sys.path.append(str(Path(__file__).resolve().parent))
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from config import SCALER_PATH, FEATURE_NAMES_PATH, RANDOM_STATE, TEST_SIZE
from logger import get_logger

logger = get_logger("preprocessor")


class ChurnPreprocessor:
    def __init__(self):
        self.scaler = RobustScaler()
        self.feature_cols = None
        self.is_fitted = False

    def prepare(self, df: pd.DataFrame, meta: dict):
        self.feature_cols = [c for c in meta["feature_cols"] if c in df.columns and c != "Churn_encoded"]
        X = df[self.feature_cols].fillna(0)
        y = df["Churn_Binary"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )

        X_train_s = pd.DataFrame(self.scaler.fit_transform(X_train), columns=self.feature_cols)
        X_test_s = pd.DataFrame(self.scaler.transform(X_test), columns=self.feature_cols)

        smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=5, n_jobs=-1)
        X_train_res, y_train_res = smote.fit_resample(X_train_s, y_train)

        joblib.dump(self.scaler, SCALER_PATH)
        joblib.dump(self.feature_cols, FEATURE_NAMES_PATH)
        self.is_fitted = True

        logger.info(f"Train: {len(X_train)} | Test: {len(X_test)} | After SMOTE: {len(X_train_res)}")
        return X_train_res, X_test_s, y_train_res, y_test

    def transform_single(self, record: dict) -> pd.DataFrame:
        if not self.is_fitted:
            self.scaler = joblib.load(SCALER_PATH)
            self.feature_cols = joblib.load(FEATURE_NAMES_PATH)
            self.is_fitted = True
        df = pd.DataFrame([record])
        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = 0
        return pd.DataFrame(
            self.scaler.transform(df[self.feature_cols].fillna(0)),
            columns=self.feature_cols
        )