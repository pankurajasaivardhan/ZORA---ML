import joblib
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent))
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from config import RANDOM_STATE, XGB_MODEL_PATH, LGBM_MODEL_PATH, RF_MODEL_PATH
from logger import get_logger

logger = get_logger("models")


class XGBoostNetworkModel:
    def __init__(self):
        self.model = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=RANDOM_STATE,
            n_jobs=-1, tree_method="hist"
        )
        self.name = "XGBoost"

    def fit(self, X, y):
        self.model.fit(X, y)
        logger.info("XGBoost training complete")
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def save(self):
        joblib.dump(self.model, XGB_MODEL_PATH)


class LightGBMNetworkModel:
    def __init__(self):
        self.model = LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            class_weight="balanced", random_state=RANDOM_STATE,
            n_jobs=-1, verbose=-1
        )
        self.name = "LightGBM"

    def fit(self, X, y):
        self.model.fit(X, y)
        logger.info("LightGBM training complete")
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def save(self):
        joblib.dump(self.model, LGBM_MODEL_PATH)


class RandomForestNetworkModel:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=200, max_depth=10,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
        )
        self.name = "RandomForest"

    def fit(self, X, y):
        self.model.fit(X, y)
        logger.info("RandomForest training complete")
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def save(self):
        joblib.dump(self.model, RF_MODEL_PATH)