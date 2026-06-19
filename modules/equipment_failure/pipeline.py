import joblib
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent))
from config import BEST_MODEL_PATH
from data_loader import EquipmentDataLoader
from preprocessor import EquipmentPreprocessor
from models import XGBoostEquipmentModel, LightGBMEquipmentModel, RandomForestEquipmentModel
from evaluator import EquipmentEvaluator
from logger import get_logger

logger = get_logger("pipeline")


class EquipmentFailurePipeline:
    def __init__(self):
        self.loader = EquipmentDataLoader()
        self.preprocessor = EquipmentPreprocessor()
        self.evaluator = EquipmentEvaluator()

    def run(self):
        logger.info("SENTINEL-ML Equipment Failure Pipeline starting")

        meta = self.loader.load_meta()
        train_df = self.loader.load_train()
        test_df = self.loader.load_test()

        X_train, X_test, y_train, y_test = self.preprocessor.prepare(train_df, test_df, meta)

        models = {
            "XGBoost": XGBoostEquipmentModel(),
            "LightGBM": LightGBMEquipmentModel(),
            "RandomForest": RandomForestEquipmentModel()
        }

        results = []
        for name, model in models.items():
            logger.info(f"Training {name}")
            if name == "XGBoost":
                model.fit(X_train, y_train, X_test, y_test)
            else:
                model.fit(X_train, y_train)
            model.save()
            y_pred = model.predict(X_test)
            result = self.evaluator.evaluate(name, y_test.values, y_pred)
            results.append(result)
        best = self.evaluator.select_best(results)
        self.evaluator.save_report(best, X_train.shape[1], len(y_test))

        best_model_obj = models[best["name"]]
        joblib.dump(
            best_model_obj.model if hasattr(best_model_obj, "model") else best_model_obj,
            BEST_MODEL_PATH
        )

        logger.info(
            f"Pipeline complete | Best={best['name']} | RMSE={best['rmse']:.4f} | "
            f"Value=${best['business_value']:,}"
        )
        return best


if __name__ == "__main__":
    pipeline = EquipmentFailurePipeline()
    pipeline.run()