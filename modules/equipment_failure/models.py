import numpy as np
import joblib
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent))
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from config import XGB_PARAMS, LGBM_PARAMS, RF_PARAMS, XGB_MODEL_PATH, LGBM_MODEL_PATH, RF_MODEL_PATH, CLIP_RUL
from logger import get_logger

logger = get_logger("models")


class XGBoostEquipmentModel:
    def __init__(self):
        self.model = XGBRegressor(**XGB_PARAMS)
        self.name = "XGBoost"

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        eval_set = [(X_val, y_val)] if X_val is not None else None
        self.model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        logger.info("XGBoost training complete")
        return self

    def predict(self, X):
        return np.clip(self.model.predict(X), 0, CLIP_RUL)

    def save(self):
        joblib.dump(self.model, XGB_MODEL_PATH)
        logger.info(f"XGBoost saved")

    def load(self):
        self.model = joblib.load(XGB_MODEL_PATH)
        return self


class LightGBMEquipmentModel:
    def __init__(self):
        self.model = LGBMRegressor(**LGBM_PARAMS)
        self.name = "LightGBM"

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        logger.info("LightGBM training complete")
        return self

    def predict(self, X):
        return np.clip(self.model.predict(X), 0, CLIP_RUL)

    def save(self):
        joblib.dump(self.model, LGBM_MODEL_PATH)

    def load(self):
        self.model = joblib.load(LGBM_MODEL_PATH)
        return self


class RandomForestEquipmentModel:
    def __init__(self):
        self.model = RandomForestRegressor(**RF_PARAMS)
        self.name = "RandomForest"

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        logger.info("RandomForest training complete")
        return self

    def predict(self, X):
        return np.clip(self.model.predict(X), 0, CLIP_RUL)

    def save(self):
        joblib.dump(self.model, RF_MODEL_PATH)

    def load(self):
        self.model = joblib.load(RF_MODEL_PATH)
        return self