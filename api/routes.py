import sys
import importlib.util
import numpy as np
import pandas as pd
from pathlib import Path
from fastapi import APIRouter, HTTPException, Depends

BASE_DIR = Path(__file__).resolve().parent.parent


def load_module(module_name, module_path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fraud_path = BASE_DIR / "modules" / "fraud"
sys.path.insert(0, str(fraud_path))

from schemas import (
    FraudTransactionRequest, FraudPredictionResponse,
    HealthResponse, PlatformStatsResponse,
    LoanApplicationRequest, LoanPredictionResponse,
    HealthRiskRequest, HealthRiskResponse,
    NetworkTrafficRequest, NetworkAnomalyResponse,
    EquipmentRiskRequest, EquipmentRiskResponse
)
from dependencies import (
    get_fraud_predictor, get_prediction_stats,
    update_prediction_stats, get_model_report,
    get_loan_predictor, get_health_predictor,
    get_equipment_predictor, get_network_predictor
)

fraud_logger_mod = load_module("fraud_logger", fraud_path / "logger.py")
logger = fraud_logger_mod.get_logger("routes")

router = APIRouter()


def build_fraud_record(request: FraudTransactionRequest) -> dict:
    raw = request.dict()
    df = pd.DataFrame([raw])
    df["hour_of_day"] = (df["Time"] / 3600) % 24
    df["day_number"] = (df["Time"] / 3600 / 24).astype(int) + 1
    df["is_night"] = ((df["hour_of_day"] >= 22) | (df["hour_of_day"] <= 5)).astype(int)
    df["is_peak_fraud_hour"] = ((df["hour_of_day"] >= 1) & (df["hour_of_day"] <= 3)).astype(int)
    df["amount_log"] = np.log1p(df["Amount"])
    df["is_zero_amount"] = (df["Amount"] == 0).astype(int)
    df["is_small_amount"] = (df["Amount"] < 10).astype(int)
    df["is_round_amount"] = (df["Amount"] % 1 == 0).astype(int)
    df["amount_above_99p"] = (df["Amount"] > 2691.0).astype(int)
    top_features = ["V14", "V3", "V17", "V12", "V10", "V11", "V4"]
    df["V14_V3_interaction"] = df["V14"] * df["V3"]
    df["V17_V12_interaction"] = df["V17"] * df["V12"]
    df["V14_V17_euclidean"] = np.sqrt(df["V14"] ** 2 + df["V17"] ** 2)
    df["top_features_sum_abs"] = df[top_features].abs().sum(axis=1)
    df["top_features_max_abs"] = df[top_features].abs().max(axis=1)
    for feat in top_features:
        df[f"{feat}_abs"] = df[feat].abs()
    for window in [10, 30, 60, 300]:
        df[f"txn_count_{window}s"] = 0
        df[f"amount_sum_{window}s"] = 0.0
    df = df.drop(columns=["Time", "Amount"], errors="ignore")
    return df.iloc[0].to_dict()


@router.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    modules = {}
    for name, loader in [
        ("fraud_detection", get_fraud_predictor),
        ("loan_default", get_loan_predictor),
        ("health_risk", get_health_predictor),
        ("equipment_failure", get_equipment_predictor),
        ("network_anomaly", get_network_predictor),
        ("customer_churn", get_churn_predictor)
    ]:
        try:
            loader()
            modules[name] = True
        except Exception:
            modules[name] = False
    modules.update({
        "equipment_failure": modules.get("equipment_failure", False),
        "network_anomaly": modules.get("network_anomaly", False),
        "customer_churn": modules.get("customer_churn", False),
        "stock_risk": False
    })
    return HealthResponse(status="operational", version="1.0.0", modules_loaded=modules)


@router.post("/predict/fraud", response_model=FraudPredictionResponse, tags=["Fraud Detection"])
def predict_fraud(request: FraudTransactionRequest, predictor=Depends(get_fraud_predictor)):
    try:
        record = build_fraud_record(request)
        result = predictor.predict(record)
        update_prediction_stats(result["is_fraud"], 200 if result["is_fraud"] else 0)
        return FraudPredictionResponse(
            fraud_score=result["fraud_score"],
            is_fraud=result["is_fraud"],
            risk_level=result["risk_level"],
            threshold_used=result["threshold_used"],
            action=result["action"]
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Fraud error: {e}")
        raise HTTPException(status_code=500, detail="Internal prediction error")


@router.post("/predict/loan-default", response_model=LoanPredictionResponse, tags=["Loan Default"])
def predict_loan_default(request: LoanApplicationRequest, predictor=Depends(get_loan_predictor)):
    try:
        result = predictor.predict(request.dict())
        return LoanPredictionResponse(
            default_score=result["default_score"],
            is_default=result["is_default"],
            risk_level=result["risk_level"],
            threshold_used=result["threshold_used"],
            decision=result["decision"],
            composite_risk_score=result["composite_risk_score"],
            grade_encoded=result["grade_encoded"]
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Loan error: {e}")
        raise HTTPException(status_code=500, detail="Internal prediction error")


@router.post("/predict/health-risk", response_model=HealthRiskResponse, tags=["Health Risk"])
def predict_health_risk(request: HealthRiskRequest, predictor=Depends(get_health_predictor)):
    try:
        data = request.dict()
        heart_record = {k: data[k] for k in [
            "age", "sex", "cp", "trestbps", "chol", "fbs",
            "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
        ]}
        diabetes_record = {k: data[k] for k in [
            "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
        ]}
        result = predictor.predict(heart_record, diabetes_record)
        return HealthRiskResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Health error: {e}")
        raise HTTPException(status_code=500, detail="Internal prediction error")


@router.get("/stats", response_model=PlatformStatsResponse, tags=["Platform"])
def platform_stats():
    stats = get_prediction_stats()
    report = get_model_report()
    return PlatformStatsResponse(
        total_predictions=stats["total"],
        fraud_detected=stats["fraud"],
        legitimate_approved=stats["legit"],
        model_name=report.get("best_model", "XGBoost"),
        model_auc=report.get("auc_roc", 0.0),
        business_value_generated=stats["business_value"]
    )


@router.get("/model/report", tags=["Platform"])
def model_report():
    return get_model_report()

from schemas import EquipmentRiskRequest, EquipmentRiskResponse
from dependencies import get_equipment_predictor


@router.post("/predict/equipment-failure", response_model=EquipmentRiskResponse, tags=["Equipment Failure"])
def predict_equipment(request: EquipmentRiskRequest, predictor=Depends(get_equipment_predictor)):
    try:
        result = predictor.predict(request.sensor_readings)
        return EquipmentRiskResponse(**result)
    except Exception as e:
        logger.error(f"Equipment error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


from schemas import NetworkTrafficRequest, NetworkAnomalyResponse
from dependencies import get_network_predictor


@router.post("/predict/network-anomaly", response_model=NetworkAnomalyResponse, tags=["Network Anomaly"])
def predict_network(request: NetworkTrafficRequest, predictor=Depends(get_network_predictor)):
    try:
        result = predictor.predict(request.dict())
        return NetworkAnomalyResponse(**result)
    except Exception as e:
        logger.error(f"Network error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


from schemas import ChurnRequest, ChurnResponse
from dependencies import get_churn_predictor


@router.post("/predict/customer-churn", response_model=ChurnResponse, tags=["Customer Churn"])
def predict_churn(request: ChurnRequest, predictor=Depends(get_churn_predictor)):
    try:
        result = predictor.predict(request.dict())
        return ChurnResponse(**result)
    except Exception as e:
        logger.error(f"Churn error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
