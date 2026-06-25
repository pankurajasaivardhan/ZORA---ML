import pandas as pd
import joblib
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent))
from config import ENGINEERED_DATA_PATH, BEST_MODEL_PATH, META_PATH
from data_loader import ChurnDataLoader
from feature_engineering import ChurnFeatureEngineer
from preprocessor import ChurnPreprocessor
from trainer import ChurnModelTrainer
from evaluator import ChurnEvaluator
from logger import get_logger
import json

logger = get_logger("pipeline")


class ChurnPipeline:
    def __init__(self):
        self.loader = ChurnDataLoader()
        self.engineer = ChurnFeatureEngineer()
        self.preprocessor = ChurnPreprocessor()
        self.trainer = ChurnModelTrainer()
        self.evaluator = ChurnEvaluator()

    def run(self):
        logger.info("SENTINEL-ML Customer Churn Pipeline starting")

        df = self.loader.load_and_clean()

        if ENGINEERED_DATA_PATH.exists():
            df_engineered = pd.read_csv(ENGINEERED_DATA_PATH)
        else:
            df_engineered = self.engineer.fit_transform(df)
            df_engineered.to_csv(ENGINEERED_DATA_PATH, index=False)

        with open(META_PATH) as f:
            meta = json.load(f)

        if "feature_cols" not in meta:
            drop_cols = ["customerID", "gender", "Partner", "Dependents", "PhoneService",
                        "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
                        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
                        "Contract", "PaperlessBilling", "PaymentMethod", "Churn", "Churn_Binary"]
            meta["feature_cols"] = [c for c in df_engineered.columns if c not in drop_cols]

        X_train, X_test, y_train, y_test = self.preprocessor.prepare(df_engineered, meta)

        models = self.trainer.train_all(X_train, y_train)

        results = []
        for name, model in models.items():
            y_prob = model.predict_proba(X_test)
            result = self.evaluator.evaluate(name, y_test, y_prob)
            results.append(result)

        xgb_result = next(r for r in results if r["name"] == "XGBoost")
        best_model_obj = models["XGBoost"]
        joblib.dump(best_model_obj.model, BEST_MODEL_PATH)

        self.evaluator.save_report(
            xgb_result, X_train.shape[1], len(y_test), int(y_test.sum()),
            meta.get("monthly_revenue_at_risk", 0)
        )

        logger.info(f"Pipeline complete | Production=XGBoost | AUC={xgb_result['auc_roc']} | Value=${xgb_result['business_value']:,}")
        return xgb_result


if __name__ == "__main__":
    pipeline = ChurnPipeline()
    pipeline.run()