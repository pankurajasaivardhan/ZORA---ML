from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

DATA_DIR = BASE_DIR / "data" / "customer_churn"
MODEL_DIR = BASE_DIR / "models" / "customer_churn"
LOG_DIR = BASE_DIR / "logs" / "customer_churn"
REPORT_DIR = BASE_DIR / "reports" / "customer_churn"

RAW_DATA_PATH = DATA_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
CLEANED_DATA_PATH = DATA_DIR / "churn_cleaned.csv"
ENGINEERED_DATA_PATH = DATA_DIR / "churn_engineered.csv"
META_PATH = DATA_DIR / "meta.json"

BEST_MODEL_PATH = MODEL_DIR / "best_model.pkl"
XGB_MODEL_PATH = MODEL_DIR / "xgb_model.pkl"
LGBM_MODEL_PATH = MODEL_DIR / "lgbm_model.pkl"
LR_MODEL_PATH = MODEL_DIR / "lr_model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
FEATURE_NAMES_PATH = MODEL_DIR / "feature_names.pkl"
MODEL_REPORT_PATH = MODEL_DIR / "model_report.json"

RANDOM_STATE = 42
TEST_SIZE = 0.2

COST_FALSE_NEGATIVE = 500
COST_FALSE_POSITIVE = 50
REWARD_TRUE_POSITIVE = 300

BINARY_MAP = {"Yes": 1, "No": 0, "Female": 1, "Male": 0}
INTERNET_MAP = {"No": 0, "DSL": 1, "Fiber optic": 2}
CONTRACT_MAP = {"Month-to-month": 0, "One year": 1, "Two year": 2}
PAYMENT_MAP = {
    "Electronic check": 0, "Mailed check": 1,
    "Bank transfer (automatic)": 2, "Credit card (automatic)": 3
}
SERVICE_MAP = {"No": 0, "Yes": 1, "No internet service": -1, "No phone service": -1}
SERVICE_COLS = ["MultipleLines", "OnlineSecurity", "OnlineBackup",
                "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]

XGB_PARAMS = {
    "n_estimators": 300, "max_depth": 5, "learning_rate": 0.05,
    "subsample": 0.8, "colsample_bytree": 0.8, "scale_pos_weight": 3,
    "eval_metric": "logloss", "random_state": RANDOM_STATE,
    "n_jobs": -1, "tree_method": "hist"
}

for path in [DATA_DIR, MODEL_DIR, LOG_DIR, REPORT_DIR]:
    path.mkdir(parents=True, exist_ok=True)