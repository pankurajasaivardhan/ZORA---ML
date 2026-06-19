import pandas as pd
import joblib
from pathlib import Path
from typing import Tuple
import sys
sys.path.append(str(Path(__file__).resolve().parent))
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from config import (
    SCALER_HEART_PATH, SCALER_DIABETES_PATH,
    HEART_FEATURES_PATH, DIABETES_FEATURES_PATH,
    RANDOM_STATE, TEST_SIZE
)
from logger import get_logger

logger = get_logger("preprocessor")


class HealthPreprocessor:
    def __init__(self):
        self.scaler_heart = RobustScaler()
        self.scaler_diabetes = RobustScaler()
        self.heart_features = None
        self.diabetes_features = None
        self.is_fitted = False

    def prepare_heart(self, df: pd.DataFrame):
        self.heart_features = [c for c in df.columns if c != "target"]
        X = df[self.heart_features]
        y = df["target"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
        X_train_s = pd.DataFrame(self.scaler_heart.fit_transform(X_train), columns=self.heart_features)
        X_test_s = pd.DataFrame(self.scaler_heart.transform(X_test), columns=self.heart_features)
        logger.info(f"Heart split — Train: {len(X_train)} | Test: {len(X_test)}")
        joblib.dump(self.scaler_heart, SCALER_HEART_PATH)
        joblib.dump(self.heart_features, HEART_FEATURES_PATH)
        return X_train_s, X_test_s, y_train, y_test

    def prepare_diabetes(self, df: pd.DataFrame):
        self.diabetes_features = [c for c in df.columns if c != "Outcome"]
        X = df[self.diabetes_features]
        y = df["Outcome"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
        X_train_s = pd.DataFrame(self.scaler_diabetes.fit_transform(X_train), columns=self.diabetes_features)
        X_test_s = pd.DataFrame(self.scaler_diabetes.transform(X_test), columns=self.diabetes_features)
        smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=5, n_jobs=-1)
        X_train_res, y_train_res = smote.fit_resample(X_train_s, y_train)
        logger.info(f"Diabetes split — Train: {len(X_train)} | Test: {len(X_test)}")
        logger.info(f"After SMOTE — Non-diabetic: {(y_train_res==0).sum()} | Diabetic: {(y_train_res==1).sum()}")
        joblib.dump(self.scaler_diabetes, SCALER_DIABETES_PATH)
        joblib.dump(self.diabetes_features, DIABETES_FEATURES_PATH)
        return X_train_res, X_test_s, y_train_res, y_test

    def transform_heart_single(self, record: dict) -> pd.DataFrame:
        scaler = joblib.load(SCALER_HEART_PATH)
        features = joblib.load(HEART_FEATURES_PATH)
        df = pd.DataFrame([record])
        missing = [f for f in features if f not in df.columns]
        for col in missing:
            df[col] = 0
        df = df[features]
        return pd.DataFrame(scaler.transform(df), columns=features)

    def transform_diabetes_single(self, record: dict) -> pd.DataFrame:
        scaler = joblib.load(SCALER_DIABETES_PATH)
        features = joblib.load(DIABETES_FEATURES_PATH)
        df = pd.DataFrame([record])
        missing = [f for f in features if f not in df.columns]
        for col in missing:
            df[col] = 0
        df = df[features]
        return pd.DataFrame(scaler.transform(df), columns=features)