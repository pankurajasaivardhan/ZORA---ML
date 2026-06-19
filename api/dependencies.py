import sys
import json
import importlib.util
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(BASE_DIR / "modules" / "fraud"))

from predictor import FraudPredictor
from config import MODEL_REPORT_PATH
from logger import get_logger

logger = get_logger("dependencies")

_fraud_predictor: FraudPredictor = None
_loan_predictor = None
_health_predictor = None
_prediction_stats = {
    "total": 0,
    "fraud": 0,
    "legit": 0,
    "business_value": 0
}


def get_fraud_predictor() -> FraudPredictor:
    global _fraud_predictor
    if _fraud_predictor is None:
        logger.info("Loading fraud predictor")
        _fraud_predictor = FraudPredictor()
        _fraud_predictor.load()
    return _fraud_predictor


def _load_module(module_prefix: str, module_dir: Path, filename: str):
    spec = importlib.util.spec_from_file_location(
        f"{module_prefix}_{filename.replace('.py', '')}",
        module_dir / filename
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def get_loan_predictor():
    global _loan_predictor
    if _loan_predictor is None:
        logger.info("Loading loan predictor")
        loan_dir = BASE_DIR / "modules" / "loan_default"
        predictor_mod = _load_module("loan", loan_dir, "predictor.py")
        _loan_predictor = predictor_mod.LoanPredictor()
        _loan_predictor.load()
    return _loan_predictor


def get_health_predictor():
    global _health_predictor
    if _health_predictor is None:
        logger.info("Loading health predictor")
        health_dir = BASE_DIR / "modules" / "health_risk"
        predictor_mod = _load_module("health", health_dir, "predictor.py")
        _health_predictor = predictor_mod.HealthRiskPredictor()
        _health_predictor.load()
    return _health_predictor


def get_prediction_stats() -> dict:
    return _prediction_stats


def update_prediction_stats(is_fraud: bool, business_value: int):
    _prediction_stats["total"] += 1
    if is_fraud:
        _prediction_stats["fraud"] += 1
    else:
        _prediction_stats["legit"] += 1
    _prediction_stats["business_value"] += business_value


def get_model_report() -> dict:
    with open(MODEL_REPORT_PATH) as f:
        return json.load(f)
_equipment_predictor = None


def get_equipment_predictor():
    global _equipment_predictor
    if _equipment_predictor is None:
        logger.info("Loading equipment predictor")
        mod = _load_module("equipment", BASE_DIR / "modules" / "equipment_failure", "predictor.py")
        _equipment_predictor = mod.EquipmentPredictor()
        _equipment_predictor.load()
    return _equipment_predictor

_network_predictor = None


def get_network_predictor():
    global _network_predictor
    if _network_predictor is None:
        logger.info("Loading network predictor")
        mod = _load_module("network", BASE_DIR / "modules" / "network_anomaly", "predictor.py")
        _network_predictor = mod.NetworkAnomalyPredictor()
        _network_predictor.load()
    return _network_predictor
