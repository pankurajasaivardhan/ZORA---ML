import joblib
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent))
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from config import XGB_PARAMS, LGBM_PARAMS, XGB_MODEL_PATH, LGBM_MODEL_PATH
from logger import get_logger

logger = get_logger("models")


class XGBoostLoanModel:
    def __init__(self, scale_pos_weight: int = 3):
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
        logger.info(f"XGBoost saved to {XGB_MODEL_PATH}")

    def load(self):
        self.model = joblib.load(XGB_MODEL_PATH)
        return self


class LightGBMLoanModel:
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
        logger.info(f"LightGBM saved to {LGBM_MODEL_PATH}")

    def load(self):
        self.model = joblib.load(LGBM_MODEL_PATH)
        return self