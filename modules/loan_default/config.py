from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

DATA_DIR = BASE_DIR / "data" / "loan_default"
MODEL_DIR = BASE_DIR / "models" / "loan_default"
LOG_DIR = BASE_DIR / "logs" / "loan_default"
REPORT_DIR = BASE_DIR / "reports" / "loan_default"

RAW_DATA_PATH = DATA_DIR / "accepted_2007_to_2018q4.csv"
CLEANED_DATA_PATH = DATA_DIR / "loan_cleaned.csv"
ENGINEERED_DATA_PATH = DATA_DIR / "loan_engineered.csv"

SCALER_PATH = MODEL_DIR / "robust_scaler.pkl"
XGB_MODEL_PATH = MODEL_DIR / "xgb_model.pkl"
LGBM_MODEL_PATH = MODEL_DIR / "lgbm_model.pkl"
LR_MODEL_PATH = MODEL_DIR / "lr_model.pkl"
BEST_MODEL_PATH = MODEL_DIR / "best_model.pkl"
FEATURE_NAMES_PATH = MODEL_DIR / "feature_names.pkl"
MODEL_REPORT_PATH = MODEL_DIR / "model_report.json"

TRAIN_RATIO = 0.8
SMOTE_SAMPLING_STRATEGY = 0.3
SMOTE_K_NEIGHBORS = 5
RANDOM_STATE = 42
CV_FOLDS = 5

XGB_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 10,
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
    "min_child_samples": 50,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbose": -1
}

COST_FALSE_NEGATIVE = 15000
COST_FALSE_POSITIVE = 500
REWARD_TRUE_POSITIVE = 8000

DEFAULT_STATUSES = [
    "Charged Off", "Default",
    "Late (31-120 days)", "Late (16-30 days)",
    "Does not meet the credit policy. Status:Charged Off"
]

GRADE_MAP = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}
HOME_MAP = {"MORTGAGE": 0, "RENT": 1, "OWN": 2, "ANY": 1, "OTHER": 1, "NONE": 1}

KEY_FEATURES = [
    "loan_amnt", "funded_amnt", "int_rate", "installment",
    "annual_inc", "dti", "delinq_2yrs", "fico_range_low",
    "fico_range_high", "open_acc", "pub_rec", "revol_bal",
    "revol_util", "total_acc", "mort_acc", "pub_rec_bankruptcies",
    "emp_length", "home_ownership", "purpose", "grade", "is_default"
]

TARGET_COLUMN = "is_default"

for path in [DATA_DIR, MODEL_DIR, LOG_DIR, REPORT_DIR]:
    path.mkdir(parents=True, exist_ok=True)

