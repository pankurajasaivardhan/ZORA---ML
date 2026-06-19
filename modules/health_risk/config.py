from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

DATA_DIR = BASE_DIR / "data" / "health_risk"
MODEL_DIR = BASE_DIR / "models" / "health_risk"
LOG_DIR = BASE_DIR / "logs" / "health_risk"
REPORT_DIR = BASE_DIR / "reports" / "health_risk"

HEART_RAW_PATH = DATA_DIR / "heart.csv"
DIABETES_RAW_PATH = DATA_DIR / "diabetes.csv"
HEART_CLEANED_PATH = DATA_DIR / "heart_cleaned.csv"
DIABETES_CLEANED_PATH = DATA_DIR / "diabetes_cleaned.csv"
HEART_ENGINEERED_PATH = DATA_DIR / "heart_engineered.csv"
DIABETES_ENGINEERED_PATH = DATA_DIR / "diabetes_engineered.csv"

HEART_MODEL_PATH = MODEL_DIR / "best_heart_model.pkl"
DIABETES_MODEL_PATH = MODEL_DIR / "best_diabetes_model.pkl"
SCALER_HEART_PATH = MODEL_DIR / "scaler_heart.pkl"
SCALER_DIABETES_PATH = MODEL_DIR / "scaler_diabetes.pkl"
HEART_FEATURES_PATH = MODEL_DIR / "heart_features.pkl"
DIABETES_FEATURES_PATH = MODEL_DIR / "diabetes_features.pkl"
MODEL_REPORT_PATH = MODEL_DIR / "model_report.json"

RANDOM_STATE = 42
CV_FOLDS = 5
TEST_SIZE = 0.2

SVM_PARAMS = {
    "kernel": "rbf", "C": 10, "gamma": "scale",
    "probability": True, "random_state": RANDOM_STATE, "class_weight": "balanced"
}

GB_PARAMS = {
    "n_estimators": 200, "max_depth": 4,
    "learning_rate": 0.05, "random_state": RANDOM_STATE
}

COST_FALSE_NEGATIVE_HEART = 50000
COST_FALSE_POSITIVE_HEART = 2000
REWARD_TRUE_POSITIVE_HEART = 30000

COST_FALSE_NEGATIVE_DIABETES = 20000
COST_FALSE_POSITIVE_DIABETES = 500
REWARD_TRUE_POSITIVE_DIABETES = 12000

HEART_WEIGHT = 0.6
DIABETES_WEIGHT = 0.4

for path in [DATA_DIR, MODEL_DIR, LOG_DIR, REPORT_DIR]:
    path.mkdir(parents=True, exist_ok=True)