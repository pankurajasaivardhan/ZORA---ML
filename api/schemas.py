from pydantic import BaseModel, Field
from typing import Dict, Optional, Any


class FraudTransactionRequest(BaseModel):
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float = Field(..., ge=0)
    Time: float = Field(..., ge=0)

    class Config:
        json_schema_extra = {
            "example": {
                "V1": -1.359807, "V2": -0.072781, "V3": 2.536347,
                "V4": 1.378155, "V5": -0.338321, "V6": 0.462388,
                "V7": 0.239599, "V8": 0.098698, "V9": 0.363787,
                "V10": 0.090794, "V11": -0.551600, "V12": -0.617801,
                "V13": -0.991390, "V14": -0.311169, "V15": 1.468177,
                "V16": -0.470401, "V17": 0.207971, "V18": 0.025791,
                "V19": 0.403993, "V20": 0.251412, "V21": -0.018307,
                "V22": 0.277838, "V23": -0.110474, "V24": 0.066928,
                "V25": 0.128539, "V26": -0.189115, "V27": 0.133558,
                "V28": -0.021053, "Amount": 149.62, "Time": 0.0
            }
        }


class FraudPredictionResponse(BaseModel):
    fraud_score: float
    is_fraud: bool
    risk_level: str
    threshold_used: float
    action: str
    explanation: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    status: str
    version: str
    modules_loaded: Dict[str, bool]


class PlatformStatsResponse(BaseModel):
    total_predictions: int
    fraud_detected: int
    legitimate_approved: int
    model_name: str
    model_auc: float
    business_value_generated: int


class LoanApplicationRequest(BaseModel):
    loan_amnt: float = Field(..., gt=0)
    funded_amnt: float = Field(..., gt=0)
    int_rate: float = Field(..., gt=0, lt=40)
    installment: float = Field(..., gt=0)
    annual_inc: float = Field(..., gt=0)
    dti: float = Field(..., ge=0)
    delinq_2yrs: float = Field(default=0, ge=0)
    fico_range_low: float = Field(..., ge=300, le=850)
    fico_range_high: float = Field(..., ge=300, le=850)
    open_acc: float = Field(default=5, ge=0)
    pub_rec: float = Field(default=0, ge=0)
    revol_bal: float = Field(default=0, ge=0)
    revol_util: float = Field(default=50, ge=0, le=100)
    total_acc: float = Field(default=10, ge=0)
    mort_acc: float = Field(default=0, ge=0)
    pub_rec_bankruptcies: float = Field(default=0, ge=0)
    emp_length: float = Field(default=5, ge=0, le=10)
    grade: str = Field(default="C")
    home_ownership: str = Field(default="RENT")
    purpose: str = Field(default="debt_consolidation")

    class Config:
        json_schema_extra = {
            "example": {
                "loan_amnt": 10000, "funded_amnt": 10000,
                "int_rate": 12.5, "installment": 250.0,
                "annual_inc": 65000, "dti": 18.5,
                "delinq_2yrs": 0, "fico_range_low": 700,
                "fico_range_high": 720, "open_acc": 8,
                "pub_rec": 0, "revol_bal": 5000,
                "revol_util": 45.0, "total_acc": 15,
                "mort_acc": 1, "pub_rec_bankruptcies": 0,
                "emp_length": 7, "grade": "B",
                "home_ownership": "MORTGAGE",
                "purpose": "debt_consolidation"
            }
        }


class LoanPredictionResponse(BaseModel):
    default_score: float
    is_default: bool
    risk_level: str
    threshold_used: float
    decision: str
    composite_risk_score: float
    grade_encoded: int


class HealthRiskRequest(BaseModel):
    age: float = Field(..., ge=18, le=100)
    sex: int = Field(..., ge=0, le=1, description="1=male 0=female")
    cp: int = Field(..., ge=0, le=3, description="Chest pain type 0-3")
    trestbps: float = Field(..., ge=80, le=220, description="Resting blood pressure")
    chol: float = Field(..., ge=100, le=600, description="Cholesterol mg/dl")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120")
    restecg: int = Field(..., ge=0, le=2)
    thalach: float = Field(..., ge=60, le=220, description="Max heart rate")
    exang: int = Field(..., ge=0, le=1, description="Exercise induced angina")
    oldpeak: float = Field(..., ge=0, le=7)
    slope: int = Field(..., ge=0, le=2)
    ca: int = Field(..., ge=0, le=4)
    thal: int = Field(..., ge=0, le=3)
    Pregnancies: float = Field(default=0, ge=0)
    Glucose: float = Field(default=120, ge=0, le=300)
    BloodPressure: float = Field(default=80, ge=0, le=200)
    SkinThickness: float = Field(default=20, ge=0)
    Insulin: float = Field(default=80, ge=0)
    BMI: float = Field(default=28.0, ge=10, le=70)
    DiabetesPedigreeFunction: float = Field(default=0.5, ge=0)
    Age: float = Field(default=35, ge=18, le=100)

    class Config:
        json_schema_extra = {
            "example": {
                "age": 55, "sex": 1, "cp": 2, "trestbps": 130,
                "chol": 250, "fbs": 0, "restecg": 1, "thalach": 160,
                "exang": 0, "oldpeak": 1.5, "slope": 1, "ca": 0, "thal": 2,
                "Pregnancies": 2, "Glucose": 120, "BloodPressure": 80,
                "SkinThickness": 20, "Insulin": 80, "BMI": 26.5,
                "DiabetesPedigreeFunction": 0.4, "Age": 55
            }
        }


class HealthRiskResponse(BaseModel):
    heart_disease_score: float
    diabetes_score: float
    combined_health_score: float
    heart_disease_detected: bool
    diabetes_detected: bool
    heart_risk_level: str
    diabetes_risk_level: str
    overall_risk_level: str
    recommendation: str

class EquipmentRiskRequest(BaseModel):
    sensor_readings: list = Field(..., description="List of sensor readings over time")

    class Config:
        json_schema_extra = {
            "example": {
                "sensor_readings": [
                    {"cycle": 1, "s2": 641.82, "s3": 1589.70, "s4": 1400.60, "s7": 554.36,
                     "s8": 2388.02, "s9": 9046.19, "s11": 47.47, "s12": 521.66,
                     "s13": 2388.02, "s14": 8138.62, "s15": 8.4195, "s17": 392,
                     "s20": 39.06, "s21": 23.4190},
                    {"cycle": 50, "s2": 642.15, "s3": 1591.82, "s4": 1403.14, "s7": 553.75,
                     "s8": 2388.08, "s9": 9044.07, "s11": 47.49, "s12": 522.28,
                     "s13": 2388.08, "s14": 8131.49, "s15": 8.4318, "s17": 392,
                     "s20": 39.00, "s21": 23.4236}
                ]
            }
        }


class EquipmentRiskResponse(BaseModel):
    predicted_rul: float
    failure_imminent: bool
    urgency_level: str
    alert_threshold: int
    health_index: float
    failure_risk_score: float
    recommendation: str


class NetworkTrafficRequest(BaseModel):
    duration: float = Field(default=0)
    protocol_type: str = Field(default="tcp")
    service: str = Field(default="http")
    flag: str = Field(default="SF")
    src_bytes: float = Field(default=0)
    dst_bytes: float = Field(default=0)
    land: float = Field(default=0)
    wrong_fragment: float = Field(default=0)
    urgent: float = Field(default=0)
    hot: float = Field(default=0)
    num_failed_logins: float = Field(default=0)
    logged_in: float = Field(default=0)
    num_compromised: float = Field(default=0)
    root_shell: float = Field(default=0)
    su_attempted: float = Field(default=0)
    num_root: float = Field(default=0)
    num_file_creations: float = Field(default=0)
    num_shells: float = Field(default=0)
    num_access_files: float = Field(default=0)
    is_host_login: float = Field(default=0)
    is_guest_login: float = Field(default=0)
    count: float = Field(default=1)
    srv_count: float = Field(default=1)
    serror_rate: float = Field(default=0)
    srv_serror_rate: float = Field(default=0)
    rerror_rate: float = Field(default=0)
    srv_rerror_rate: float = Field(default=0)
    same_srv_rate: float = Field(default=1)
    diff_srv_rate: float = Field(default=0)
    srv_diff_host_rate: float = Field(default=0)
    dst_host_count: float = Field(default=1)
    dst_host_srv_count: float = Field(default=1)
    dst_host_same_srv_rate: float = Field(default=1)
    dst_host_diff_srv_rate: float = Field(default=0)
    dst_host_same_src_port_rate: float = Field(default=0)
    dst_host_srv_diff_host_rate: float = Field(default=0)
    dst_host_serror_rate: float = Field(default=0)
    dst_host_srv_serror_rate: float = Field(default=0)
    dst_host_rerror_rate: float = Field(default=0)
    dst_host_srv_rerror_rate: float = Field(default=0)
    service_attack_rate: float = Field(default=0.465)
    service_frequency: float = Field(default=1000)

    class Config:
        json_schema_extra = {
            "example": {
                "duration": 0, "protocol_type": "tcp", "service": "http",
                "flag": "SF", "src_bytes": 181, "dst_bytes": 5450,
                "land": 0, "wrong_fragment": 0, "urgent": 0, "hot": 0,
                "num_failed_logins": 0, "logged_in": 1, "num_compromised": 0,
                "root_shell": 0, "su_attempted": 0, "num_root": 0,
                "num_file_creations": 0, "num_shells": 0, "num_access_files": 0,
                "is_host_login": 0, "is_guest_login": 0, "count": 8,
                "srv_count": 8, "serror_rate": 0.0, "srv_serror_rate": 0.0,
                "rerror_rate": 0.0, "srv_rerror_rate": 0.0, "same_srv_rate": 1.0,
                "diff_srv_rate": 0.0, "srv_diff_host_rate": 0.0,
                "dst_host_count": 9, "dst_host_srv_count": 9,
                "dst_host_same_srv_rate": 1.0, "dst_host_diff_srv_rate": 0.0,
                "dst_host_same_src_port_rate": 0.11, "dst_host_srv_diff_host_rate": 0.0,
                "dst_host_serror_rate": 0.0, "dst_host_srv_serror_rate": 0.0,
                "dst_host_rerror_rate": 0.0, "dst_host_srv_rerror_rate": 0.0,
                "service_attack_rate": 0.1, "service_frequency": 5000
            }
        }


class NetworkAnomalyResponse(BaseModel):
    attack_score: float
    is_attack: bool
    threat_level: str
    threshold_used: float
    action: str
    network_risk_score: float
    recommendation: str


class ChurnRequest(BaseModel):
    gender: str = Field(default="Female")
    SeniorCitizen: int = Field(default=0, ge=0, le=1)
    Partner: str = Field(default="No")
    Dependents: str = Field(default="No")
    tenure: float = Field(default=12, ge=0, le=100)
    PhoneService: str = Field(default="Yes")
    MultipleLines: str = Field(default="No")
    InternetService: str = Field(default="Fiber optic")
    OnlineSecurity: str = Field(default="No")
    OnlineBackup: str = Field(default="No")
    DeviceProtection: str = Field(default="No")
    TechSupport: str = Field(default="No")
    StreamingTV: str = Field(default="No")
    StreamingMovies: str = Field(default="No")
    Contract: str = Field(default="Month-to-month")
    PaperlessBilling: str = Field(default="Yes")
    PaymentMethod: str = Field(default="Electronic check")
    MonthlyCharges: float = Field(default=70.0, ge=0)
    TotalCharges: float = Field(default=840.0, ge=0)

    class Config:
        json_schema_extra = {
            "example": {
                "gender": "Female", "SeniorCitizen": 0, "Partner": "No",
                "Dependents": "No", "tenure": 5, "PhoneService": "Yes",
                "MultipleLines": "No", "InternetService": "Fiber optic",
                "OnlineSecurity": "No", "OnlineBackup": "No",
                "DeviceProtection": "No", "TechSupport": "No",
                "StreamingTV": "Yes", "StreamingMovies": "Yes",
                "Contract": "Month-to-month", "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 95.0, "TotalCharges": 475.0
            }
        }


class ChurnResponse(BaseModel):
    churn_score: float
    will_churn: bool
    risk_level: str
    threshold_used: float
    action: str
    churn_risk_score: float
    recommendation: str
