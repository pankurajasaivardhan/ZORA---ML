import joblib
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent))
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from config import SVM_PARAMS, GB_PARAMS, HEART_MODEL_PATH, DIABETES_MODEL_PATH
from logger import get_logger

logger = get_logger("models")


class SVMHealthModel:
    def __init__(self):
        self.model = SVC(**SVM_PARAMS)
        self.name = "SVM"

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        logger.info(f"SVM training complete")
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


class GradientBoostingHealthModel:
    def __init__(self):
        self.model = GradientBoostingClassifier(**GB_PARAMS)
        self.name = "GradientBoosting"

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        logger.info(f"GradientBoosting training complete")
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def save(self, path):
        joblib.dump(self.model, path)
        logger.info(f"Model saved to {path}")

    def load(self, path):
        self.model = joblib.load(path)
        return self