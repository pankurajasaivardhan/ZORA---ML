import json
import importlib.util
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict

BASE_DIR = Path(__file__).resolve().parent


def _load_local(filename: str):
    spec = importlib.util.spec_from_file_location(
        f"equip_local_{filename.replace('.py', '')}",
        BASE_DIR / filename
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_config = _load_local("config.py")
_logger_mod = _load_local("logger.py")

BEST_MODEL_PATH = _config.BEST_MODEL_PATH
SCALER_PATH = _config.SCALER_PATH
FEATURE_NAMES_PATH = _config.FEATURE_NAMES_PATH
MODEL_REPORT_PATH = _config.MODEL_REPORT_PATH
FAILURE_ALERT_THRESHOLD = _config.FAILURE_ALERT_THRESHOLD
CLIP_RUL = _config.CLIP_RUL
WINDOW_SIZES = _config.WINDOW_SIZES

logger = _logger_mod.get_logger("equipment_predictor")


class EquipmentPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.alert_threshold = FAILURE_ALERT_THRESHOLD
        self._loaded = False

    def load(self):
        self.model = joblib.load(BEST_MODEL_PATH)
        self.scaler = joblib.load(SCALER_PATH)
        self.feature_names = joblib.load(FEATURE_NAMES_PATH)
        with open(MODEL_REPORT_PATH) as f:
            report = json.load(f)
        self.alert_threshold = report.get("alert_threshold", FAILURE_ALERT_THRESHOLD)
        self._loaded = True
        logger.info(f"Equipment predictor loaded | alert_threshold={self.alert_threshold}")
        return self

    def _engineer_features(self, sensor_readings: list) -> dict:
        if len(sensor_readings) < 1:
            raise ValueError("At least 1 sensor reading required")

        df = pd.DataFrame(sensor_readings)
        useful_sensors = [c for c in df.columns if c.startswith("s") and c[1:].isdigit()]

        for sensor in useful_sensors:
            for w in WINDOW_SIZES:
                df[f"{sensor}_roll_mean_{w}"] = df[sensor].rolling(w, min_periods=1).mean()
                df[f"{sensor}_roll_std_{w}"] = df[sensor].rolling(w, min_periods=1).std().fillna(0)
                df[f"{sensor}_roll_min_{w}"] = df[sensor].rolling(w, min_periods=1).min()
                df[f"{sensor}_roll_max_{w}"] = df[sensor].rolling(w, min_periods=1).max()

        for sensor in useful_sensors:
            for lag in [1, 2, 5]:
                df[f"{sensor}_lag_{lag}"] = df[sensor].shift(lag).fillna(method="bfill")
                df[f"{sensor}_diff_{lag}"] = df[sensor] - df[f"{sensor}_lag_{lag}"]

        max_cycle = df["cycle"].max() if "cycle" in df.columns else len(df)
        df["cycle_normalized"] = df["cycle"] / max_cycle if "cycle" in df.columns else np.linspace(0, 1, len(df))

        sensor_vals = df[useful_sensors]
        sensor_min = sensor_vals.min()
        sensor_max = sensor_vals.max()
        sensor_range = sensor_max - sensor_min + 1e-10
        normalized = (sensor_vals - sensor_min) / sensor_range
        df["health_index"] = 1 - normalized.mean(axis=1)
        df["sensor_mean"] = sensor_vals.mean(axis=1)
        df["sensor_std"] = sensor_vals.std(axis=1)
        df["sensor_max"] = sensor_vals.max(axis=1)
        df["sensor_min"] = sensor_vals.min(axis=1)
        df["sensor_range"] = df["sensor_max"] - df["sensor_min"]
        df["anomaly_score"] = ((sensor_vals - sensor_min) / sensor_range).max(axis=1)
        df["failure_risk"] = df["cycle_normalized"] * 0.4 + (1 - df["health_index"]) * 0.6

        return df.iloc[-1].to_dict()

    def predict(self, sensor_readings: list) -> Dict:
        if not self._loaded:
            self.load()

        engineered = self._engineer_features(sensor_readings)
        df = pd.DataFrame([engineered])
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0
        df = df[self.feature_names].fillna(0)
        df_scaled = pd.DataFrame(self.scaler.transform(df), columns=self.feature_names)

        rul_pred = float(np.clip(self.model.predict(df_scaled)[0], 0, CLIP_RUL))
        failure_imminent = rul_pred <= self.alert_threshold
        health_index = float(engineered.get("health_index", 0.5))
        failure_risk = float(engineered.get("failure_risk", 0.5))

        urgency = (
            "CRITICAL" if rul_pred <= 10 else
            "HIGH" if rul_pred <= self.alert_threshold else
            "MEDIUM" if rul_pred <= 60 else
            "LOW"
        )

        result = {
            "predicted_rul": round(rul_pred, 2),
            "failure_imminent": failure_imminent,
            "urgency_level": urgency,
            "alert_threshold": self.alert_threshold,
            "health_index": round(health_index, 4),
            "failure_risk_score": round(failure_risk, 4),
            "recommendation": (
                "STOP IMMEDIATELY — Critical failure risk" if urgency == "CRITICAL" else
                "SCHEDULE MAINTENANCE NOW" if urgency == "HIGH" else
                "MONITOR CLOSELY — Plan maintenance" if urgency == "MEDIUM" else
                "OPERATING NORMALLY"
            )
        }
        logger.info(f"Equipment prediction | RUL={rul_pred:.2f} | urgency={urgency}")
        return result