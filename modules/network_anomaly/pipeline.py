import joblib
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent))
from config import BEST_MODEL_PATH
from data_loader import NetworkDataLoader
from preprocessor import NetworkPreprocessor
from models import XGBoostNetworkModel, LightGBMNetworkModel, RandomForestNetworkModel
from evaluator import NetworkEvaluator
from logger import get_logger

logger = get_logger("pipeline")


class NetworkAnomalyPipeline:
    def __init__(self):
        self.loader = NetworkDataLoader()
        self.preprocessor = NetworkPreprocessor()
        self.evaluator = NetworkEvaluator()

    def run(self):
        logger.info("SENTINEL-ML Network Anomaly Pipeline starting")

        meta = self.loader.load_meta()
        train_df, y_train = self.loader.load_train()
        test_df, y_test = self.loader.load_test()

        X_train, X_test, y_train, y_test = self.preprocessor.prepare(
            train_df, y_train, test_df, y_test, meta
        )

        models = {
            "XGBoost": XGBoostNetworkModel(),
            "LightGBM": LightGBMNetworkModel(),
            "RandomForest": RandomForestNetworkModel()
        }

        results = []
        for name, model in models.items():
            logger.info(f"Training {name}")
            model.fit(X_train, y_train)
            model.save()
            y_prob = model.predict_proba(X_test)
            result = self.evaluator.evaluate(name, y_test, y_prob)
            results.append(result)

        best = self.evaluator.select_best(results)
        self.evaluator.save_report(best, X_train.shape[1], len(y_test), int(y_test.sum()))

        best_model_obj = models[best["name"]]
        joblib.dump(best_model_obj.model, BEST_MODEL_PATH)

        logger.info(f"Pipeline complete | Best={best['name']} | AUC={best['auc_roc']} | Value=${best['business_value']:,}")
        return best


if __name__ == "__main__":
    pipeline = NetworkAnomalyPipeline()
    pipeline.run()