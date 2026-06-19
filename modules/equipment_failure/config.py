from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

DATA_DIR = BASE_DIR / "data" / "equipment_failure"
MODEL_DIR = BASE_DIR / "models" / "equipment_failure"
LOG_DIR = BASE_DIR / "logs" / "equipment_failure"
REPORT_DIR = BASE_DIR / "reports" / "equipment_failure"

TRAIN_ENGINEERED_PATH = DATA_DIR / "train_engineered.csv"
TEST_ENGINEERED_PATH = DATA_DIR / "test_engineered.csv"
META_PATH = DATA_DIR / "meta.json"

BEST_MODEL_PATH = MODEL_DIR / "best_model.pkl"
XGB_MODEL_PATH = MODEL_DIR / "xgb_model.pkl"
LGBM_MODEL_PATH = MODEL_DIR / "lgbm_model.pkl"
RF_MODEL_PATH = MODEL_DIR / "rf_model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
FEATURE_NAMES_PATH = MODEL_DIR / "feature_names.pkl"
MODEL_REPORT_PATH = MODEL_DIR / "model_report.json"

CLIP_RUL = 125
FAILURE_ALERT_THRESHOLD = 30
WINDOW_SIZES = [5, 10, 20, 30]
LAG_SIZES = [1, 2, 5]
RANDOM_STATE = 42

COST_FALSE_ALARM = 5000
COST_MISSED_FAILURE = 500000
REWARD_CAUGHT_FAILURE = 50000

XGB_PARAMS = {
    "n_estimators": 500, "max_depth": 6, "learning_rate": 0.05,
    "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5,
    "gamma": 0.1, "reg_alpha": 0.1, "reg_lambda": 1.0,
    "random_state": RANDOM_STATE, "n_jobs": -1, "tree_method": "hist"
}

LGBM_PARAMS = {
    "n_estimators": 500, "max_depth": 6, "learning_rate": 0.05,
    "num_leaves": 31, "subsample": 0.8, "colsample_bytree": 0.8,
    "reg_alpha": 0.1, "reg_lambda": 1.0,
    "random_state": RANDOM_STATE, "n_jobs": -1, "verbose": -1
}

RF_PARAMS = {
    "n_estimators": 200, "max_depth": 10,
    "min_samples_leaf": 5, "random_state": RANDOM_STATE, "n_jobs": -1
}

for path in [DATA_DIR, MODEL_DIR, LOG_DIR, REPORT_DIR]:
    path.mkdir(parents=True, exist_ok=True)