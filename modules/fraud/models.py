import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Any
import sys
sys.path.append(str(Path(__file__).resolve().parent))
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import IsolationForest
from config import XGB_PARAMS, LGBM_PARAMS, ISO_PARAMS, XGB_MODEL_PATH, LGBM_MODEL_PATH, ISO_MODEL_PATH
from logger import get_logger

logger = get_logger("models")


class XGBoostFraudModel:
    def __init__(self, scale_pos_weight: int = 10):
        params = XGB_PARAMS.copy()
        params["scale_pos_weight"] = scale_pos_weight
        self.model = XGBClassifier(**params)
        self.name = "XGBoost"

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        eval_set = [(X_val, y_val)] if X_val is not None else None
        self.model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        logger.info("XGBoost training complete")
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X, threshold: float = 0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def save(self):
        joblib.dump(self.model, XGB_MODEL_PATH)
        logger.info(f"XGBoost model saved to {XGB_MODEL_PATH}")

    def load(self):
        self.model = joblib.load(XGB_MODEL_PATH)
        logger.info("XGBoost model loaded")
        return self


class LightGBMFraudModel:
    def __init__(self):
        self.model = LGBMClassifier(**LGBM_PARAMS)
        self.name = "LightGBM"

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        logger.info("LightGBM training complete")
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X, threshold: float = 0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def save(self):
        joblib.dump(self.model, LGBM_MODEL_PATH)
        logger.info(f"LightGBM model saved to {LGBM_MODEL_PATH}")

    def load(self):
        self.model = joblib.load(LGBM_MODEL_PATH)
        logger.info("LightGBM model loaded")
        return self


class IsolationForestFraudModel:
    def __init__(self, contamination: float = 0.002):
        params = ISO_PARAMS.copy()
        params["contamination"] = contamination
        self.model = IsolationForest(**params)
        self.name = "IsolationForest"
        self._score_min = None
        self._score_max = None

    def fit(self, X_train):
        self.model.fit(X_train)
        scores = self.model.decision_function(X_train)
        self._score_min = scores.min()
        self._score_max = scores.max()
        logger.info("Isolation Forest training complete")
        return self

    def predict_proba(self, X):
        scores = self.model.decision_function(X)
        return 1 - (scores - self._score_min) / (self._score_max - self._score_min + 1e-10)

    def predict(self, X, threshold: float = 0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def save(self):
        joblib.dump({"model": self.model, "score_min": self._score_min, "score_max": self._score_max}, ISO_MODEL_PATH)
        logger.info(f"Isolation Forest saved to {ISO_MODEL_PATH}")

    def load(self):
        data = joblib.load(ISO_MODEL_PATH)
        self.model = data["model"]
        self._score_min = data["score_min"]
        self._score_max = data["score_max"]
        logger.info("Isolation Forest loaded")
        return self