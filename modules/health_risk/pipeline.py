import pandas as pd
import joblib
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent))
from config import (
    HEART_ENGINEERED_PATH, DIABETES_ENGINEERED_PATH,
    HEART_MODEL_PATH, DIABETES_MODEL_PATH
)
from data_loader import HealthDataLoader
from feature_engineering import HealthFeatureEngineer
from preprocessor import HealthPreprocessor
from trainer import HealthModelTrainer
from evaluator import HealthEvaluator
from config import (
    COST_FALSE_NEGATIVE_HEART, COST_FALSE_POSITIVE_HEART, REWARD_TRUE_POSITIVE_HEART,
    COST_FALSE_NEGATIVE_DIABETES, COST_FALSE_POSITIVE_DIABETES, REWARD_TRUE_POSITIVE_DIABETES
)
from logger import get_logger

logger = get_logger("pipeline")


class HealthRiskPipeline:
    def __init__(self):
        self.loader = HealthDataLoader()
        self.engineer = HealthFeatureEngineer()
        self.preprocessor = HealthPreprocessor()
        self.trainer = HealthModelTrainer()
        self.evaluator = HealthEvaluator()

    def run(self):
        logger.info("SENTINEL-ML Health Risk Pipeline starting")

        heart_raw = self.loader.load_heart()
        diabetes_raw = self.loader.load_diabetes()

        if HEART_ENGINEERED_PATH.exists():
            heart_eng = pd.read_csv(HEART_ENGINEERED_PATH)
        else:
            heart_eng = self.engineer.engineer_heart(heart_raw)
            heart_eng.to_csv(HEART_ENGINEERED_PATH, index=False)

        if DIABETES_ENGINEERED_PATH.exists():
            diabetes_eng = pd.read_csv(DIABETES_ENGINEERED_PATH)
        else:
            diabetes_eng = self.engineer.engineer_diabetes(diabetes_raw)
            diabetes_eng.to_csv(DIABETES_ENGINEERED_PATH, index=False)

        X_h_train, X_h_test, y_h_train, y_h_test = self.preprocessor.prepare_heart(heart_eng)
        X_d_train, X_d_test, y_d_train, y_d_test = self.preprocessor.prepare_diabetes(diabetes_eng)

        heart_models = self.trainer.train_heart(X_h_train, y_h_train)
        diabetes_models = self.trainer.train_diabetes(X_d_train, y_d_train)

        heart_results = []
        for name, model in heart_models.items():
            y_prob = model.predict_proba(X_h_test)
            result = self.evaluator.evaluate(
                name, y_h_test, y_prob,
                COST_FALSE_NEGATIVE_HEART, COST_FALSE_POSITIVE_HEART, REWARD_TRUE_POSITIVE_HEART
            )
            heart_results.append(result)

        diabetes_results = []
        for name, model in diabetes_models.items():
            y_prob = model.predict_proba(X_d_test)
            result = self.evaluator.evaluate(
                name, y_d_test, y_prob,
                COST_FALSE_NEGATIVE_DIABETES, COST_FALSE_POSITIVE_DIABETES, REWARD_TRUE_POSITIVE_DIABETES
            )
            diabetes_results.append(result)

        best_heart = max(heart_results, key=lambda x: x["auc_roc"])
        best_diabetes = max(diabetes_results, key=lambda x: x["auc_roc"])

        self.trainer.save_best(
            heart_models[best_heart["name"]],
            diabetes_models[best_diabetes["name"]]
        )

        self.evaluator.save_report(
            best_heart, best_diabetes,
            X_h_train.shape[1], X_d_train.shape[1],
            len(y_h_test), len(y_d_test)
        )

        logger.info(f"Heart best: {best_heart['name']} | AUC={best_heart['auc_roc']} | Value=${best_heart['business_value']:,}")
        logger.info(f"Diabetes best: {best_diabetes['name']} | AUC={best_diabetes['auc_roc']} | Value=${best_diabetes['business_value']:,}")
        logger.info("Health Risk Pipeline complete")

        return best_heart, best_diabetes


if __name__ == "__main__":
    pipeline = HealthRiskPipeline()
    pipeline.run()