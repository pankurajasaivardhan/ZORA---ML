import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

DATA_DIR = BASE_DIR / "data" / "fraud"
MODEL_DIR = BASE_DIR / "models" / "fraud"
LOG_DIR = BASE_DIR / "logs" / "fraud"
REPORT_DIR = BASE_DIR / "reports" / "fraud"

RAW_DATA_PATH = DATA_DIR / "creditcard.csv"
ENGINEERED_DATA_PATH = DATA_DIR / "creditcard_engineered.csv"

SCALER_PATH = MODEL_DIR / "robust_scaler.pkl"
XGB_MODEL_PATH = MODEL_DIR / "xgb_fraud_model.pkl"
LGBM_MODEL_PATH = MODEL_DIR / "lgbm_fraud_model.pkl"
ISO_MODEL_PATH = MODEL_DIR / "iso_fraud_model.pkl"
BEST_MODEL_PATH = MODEL_DIR / "best_fraud_model.pkl"
FEATURE_NAMES_PATH = MODEL_DIR / "feature_names.pkl"
MODEL_REPORT_PATH = MODEL_DIR / "model_report.json"

TRAIN_RATIO = 0.8
SMOTE_SAMPLING_STRATEGY = 0.1
SMOTE_K_NEIGHBORS = 5
RANDOM_STATE = 42
CV_FOLDS = 5

XGB_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "gamma": 1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "eval_metric": "aucpr",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "tree_method": "hist"
}

LGBM_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 20,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbose": -1
}

ISO_PARAMS = {
    "n_estimators": 300,
    "max_samples": "auto",
    "max_features": 1.0,
    "bootstrap": True,
    "random_state": RANDOM_STATE,
    "n_jobs": -1
}

COST_FALSE_NEGATIVE = 500
COST_FALSE_POSITIVE = 10
REWARD_TRUE_POSITIVE = 200

TOP_PCA_FEATURES = ["V14", "V3", "V17", "V12", "V10", "V11", "V4"]
VELOCITY_WINDOWS = [10, 30, 60, 300]

TARGET_COLUMN = "Class"
FRAUD_LABEL = 1
LEGIT_LABEL = 0

for path in [DATA_DIR, MODEL_DIR, LOG_DIR, REPORT_DIR]:
    path.mkdir(parents=True, exist_ok=True)