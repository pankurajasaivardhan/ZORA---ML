from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

DATA_DIR = BASE_DIR / "data" / "network_anomaly"
MODEL_DIR = BASE_DIR / "models" / "network_anomaly"
LOG_DIR = BASE_DIR / "logs" / "network_anomaly"
REPORT_DIR = BASE_DIR / "reports" / "network_anomaly"

TRAIN_CLEANED_PATH = DATA_DIR / "train_cleaned.csv"
TEST_CLEANED_PATH = DATA_DIR / "test_cleaned.csv"
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

RANDOM_STATE = 42
COST_FALSE_NEGATIVE = 100000
COST_FALSE_POSITIVE = 500
REWARD_TRUE_POSITIVE = 50000

PROTOCOL_MAP = {"tcp": 0, "udp": 1, "icmp": 2}
FLAG_MAP = {
    "SF": 0, "S0": 1, "REJ": 2, "RSTR": 3, "SH": 4,
    "RSTO": 5, "S1": 6, "RSTOS0": 7, "S3": 8, "S2": 9, "OTH": 10
}

for path in [DATA_DIR, MODEL_DIR, LOG_DIR, REPORT_DIR]:
    path.mkdir(parents=True, exist_ok=True)