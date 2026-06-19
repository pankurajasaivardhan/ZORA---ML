import pandas as pd
import joblib
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent))
from config import ENGINEERED_DATA_PATH, BEST_MODEL_PATH
from data_loader import LoanDataLoader
from feature_engineering import LoanFeatureEngineer
from preprocessor import LoanPreprocessor
from trainer import LoanModelTrainer
from evaluator import LoanEvaluator
from explainer import LoanExplainer
from logger import get_logger

logger = get_logger("pipeline")


class LoanDefaultPipeline:
    def __init__(self):
        self.loader = LoanDataLoader()
        self.engineer = LoanFeatureEngineer()
        self.preprocessor = LoanPreprocessor()
        self.trainer = LoanModelTrainer()
        self.evaluator = LoanEvaluator()

    def run(self):
        logger.info("SENTINEL-ML Loan Default Pipeline starting")

        df = self.loader.load_and_prepare()

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

        models = self.trainer.train_all(X_train, y_train, X_test, y_test)

        results = []
        for key, model in models.items():
            y_prob = model.predict_proba(X_test)
            result = self.evaluator.evaluate(model.name, y_test, y_prob)
            results.append(result)

        best = self.evaluator.select_best(results)
        self.evaluator.save_report(best, X_train.shape[1], len(y_test), int(y_test.sum()))

        best_model_obj = models["xgb"] if best["name"] == "XGBoost" else models["lgbm"]
        joblib.dump(best_model_obj.model, BEST_MODEL_PATH)

        explainer = LoanExplainer(best_model_obj.model)
        explainer.compute_shap(X_test)
        explainer.plot_summary(X_test)

        logger.info(
            f"Pipeline complete | Best={best['name']} | AUC={best['auc_roc']} | "
            f"Caught={best['tp']} | Missed={best['fn']} | Value=${best['business_value']:,}"
        )
        return best


if __name__ == "__main__":
    pipeline = LoanDefaultPipeline()
    pipeline.run()