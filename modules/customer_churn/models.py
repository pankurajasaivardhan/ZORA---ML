import joblib
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent))
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from config import XGB_PARAMS, RANDOM_STATE, XGB_MODEL_PATH, LGBM_MODEL_PATH, LR_MODEL_PATH
from logger import get_logger

logger = get_logger("models")


class XGBoostChurnModel:
    def __init__(self):
        self.model = XGBClassifier(**XGB_PARAMS)
        self.name = "XGBoost"

    def fit(self, X, y):
        self.model.fit(X, y)
        logger.info("XGBoost training complete")
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def save(self):
        joblib.dump(self.model, XGB_MODEL_PATH)


class LightGBMChurnModel:
    def __init__(self):
        self.model = LGBMClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
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


class LogisticRegressionChurnModel:
    def __init__(self):
        self.model = LogisticRegression(
            max_iter=1000, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1, C=0.5
        )
        self.name = "LogisticReg"

    def fit(self, X, y):
        self.model.fit(X, y)
        logger.info("LogisticRegression training complete")
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def save(self):
        joblib.dump(self.model, LR_MODEL_PATH)