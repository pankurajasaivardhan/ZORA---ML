import joblib
from pathlib import Path
from typing import Dict
import sys
sys.path.append(str(Path(__file__).resolve().parent))
from models import XGBoostChurnModel, LightGBMChurnModel, LogisticRegressionChurnModel
from config import FEATURE_NAMES_PATH
from logger import get_logger

logger = get_logger("trainer")


class ChurnModelTrainer:

    def train_all(self, X_train, y_train) -> Dict:
        models = {
            "XGBoost": XGBoostChurnModel(),
            "LightGBM": LightGBMChurnModel(),
            "LogisticReg": LogisticRegressionChurnModel()
        }
        trained = {}
        for name, model in models.items():
            logger.info(f"Training {name}")
            model.fit(X_train, y_train)
            model.save()
            trained[name] = model
        return trained