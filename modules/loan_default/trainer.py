import joblib
from pathlib import Path
from typing import Dict
import sys
sys.path.append(str(Path(__file__).resolve().parent))
from sklearn.model_selection import StratifiedKFold, cross_val_score
from config import CV_FOLDS, RANDOM_STATE, FEATURE_NAMES_PATH
from models import XGBoostLoanModel, LightGBMLoanModel
from logger import get_logger

logger = get_logger("trainer")


class LoanModelTrainer:
    def __init__(self):
        self.xgb = None
        self.lgbm = None

    def _cross_validate(self, model, X, y, name: str) -> float:
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_val_score(model.model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
        logger.info(f"{name} CV AUC: {scores.mean():.4f} (+/- {scores.std():.4f})")
        return scores.mean()

    def train_all(self, X_train, y_train, X_test, y_test) -> Dict:
        scale_pos = int((y_train == 0).sum() / (y_train == 1).sum())

        logger.info("Training XGBoost")
        self.xgb = XGBoostLoanModel(scale_pos_weight=scale_pos)
        self._cross_validate(self.xgb, X_train, y_train, "XGBoost")
        self.xgb.fit(X_train, y_train, X_test, y_test)
        self.xgb.save()

        logger.info("Training LightGBM")
        self.lgbm = LightGBMLoanModel()
        self._cross_validate(self.lgbm, X_train, y_train, "LightGBM")
        self.lgbm.fit(X_train, y_train)
        self.lgbm.save()

        joblib.dump(list(X_train.columns), FEATURE_NAMES_PATH)
        logger.info("All models trained and saved")
        return {"xgb": self.xgb, "lgbm": self.lgbm}