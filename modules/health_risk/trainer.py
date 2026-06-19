import joblib
from pathlib import Path
from typing import Dict
import sys
sys.path.append(str(Path(__file__).resolve().parent))
from sklearn.model_selection import StratifiedKFold, cross_val_score
from config import CV_FOLDS, RANDOM_STATE, HEART_MODEL_PATH, DIABETES_MODEL_PATH
from models import SVMHealthModel, GradientBoostingHealthModel
from logger import get_logger

logger = get_logger("trainer")


class HealthModelTrainer:

    def _cross_validate(self, model, X, y, name: str) -> float:
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_val_score(model.model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
        logger.info(f"{name} CV AUC: {scores.mean():.4f} (+/- {scores.std():.4f})")
        return scores.mean()

    def train_heart(self, X_train, y_train) -> Dict:
        models = {
            "SVM": SVMHealthModel(),
            "GradientBoosting": GradientBoostingHealthModel()
        }
        trained = {}
        for name, model in models.items():
            logger.info(f"Training heart {name}")
            self._cross_validate(model, X_train, y_train, name)
            model.fit(X_train, y_train)
            trained[name] = model
        return trained

    def train_diabetes(self, X_train, y_train) -> Dict:
        models = {
            "SVM": SVMHealthModel(),
            "GradientBoosting": GradientBoostingHealthModel()
        }
        trained = {}
        for name, model in models.items():
            logger.info(f"Training diabetes {name}")
            self._cross_validate(model, X_train, y_train, name)
            model.fit(X_train, y_train)
            trained[name] = model
        return trained

    def save_best(self, heart_model, diabetes_model):
        joblib.dump(
            heart_model.model if hasattr(heart_model, "model") else heart_model,
            HEART_MODEL_PATH
        )
        joblib.dump(
            diabetes_model.model if hasattr(diabetes_model, "model") else diabetes_model,
            DIABETES_MODEL_PATH
        )
        logger.info("Best heart and diabetes models saved")