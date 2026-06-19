import pandas as pd
import joblib
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent))
from config import ENGINEERED_DATA_PATH, BEST_MODEL_PATH, TARGET_COLUMN
from data_loader import FraudDataLoader
from preprocessor import FraudPreprocessor
from feature_engineering import FraudFeatureEngineer
from trainer import FraudModelTrainer
from evaluator import FraudEvaluator
from explainer import FraudExplainer
from logger import get_logger

logger = get_logger("pipeline")


class FraudDetectionPipeline:
    def __init__(self):
        self.loader = FraudDataLoader()
        self.engineer = FraudFeatureEngineer()
        self.preprocessor = FraudPreprocessor()
        self.trainer = FraudModelTrainer()
        self.evaluator = FraudEvaluator()

    def run(self):
        logger.info("SENTINEL-ML Fraud Detection Pipeline starting")

        df = self.loader.load_and_validate()

        if ENGINEERED_DATA_PATH.exists():
            logger.info("Loading pre-engineered dataset")
            df_engineered = pd.read_csv(ENGINEERED_DATA_PATH)
        else:
            logger.info("Running feature engineering")
            df_engineered = self.engineer.fit_transform(df)
            df_engineered.to_csv(ENGINEERED_DATA_PATH, index=False)

        df_train, df_test = self.preprocessor.temporal_split(df_engineered)
        X_train, y_train = self.preprocessor.fit_transform(df_train)
        X_test, y_test = self.preprocessor.transform(df_test)

        contamination = df_train[TARGET_COLUMN].mean()
        models = self.trainer.train_all(X_train, y_train, X_test, y_test, contamination)

        results = []
        for key, model in models.items():
            if key == "iso":
                y_prob = model.predict_proba(X_test)
            else:
                y_prob = model.predict_proba(X_test)
            result = self.evaluator.evaluate(model.name, y_test, y_prob)
            results.append(result)

        best = self.evaluator.select_best(results)
        self.evaluator.save_report(best, X_train.shape[1], len(y_test), int(y_test.sum()))

        best_model_obj = models["xgb"] if best["name"] == "XGBoost" else (
            models["lgbm"] if best["name"] == "LightGBM" else models["iso"]
        )
        joblib.dump(best_model_obj.model if hasattr(best_model_obj, "model") else best_model_obj, BEST_MODEL_PATH)

        explainer = FraudExplainer(best_model_obj.model if hasattr(best_model_obj, "model") else best_model_obj)
        explainer.compute_shap(X_test)
        explainer.plot_summary(X_test)

        logger.info(
            f"Pipeline complete | Best={best['name']} | AUC={best['auc_roc']} | "
            f"Caught={best['tp']} | Missed={best['fn']} | BusinessValue=${best['business_value']:,}"
        )
        return best


if __name__ == "__main__":
    pipeline = FraudDetectionPipeline()
    pipeline.run()