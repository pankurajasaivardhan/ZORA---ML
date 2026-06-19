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
        f"network_local_{filename.replace('.py', '')}",
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
PROTOCOL_MAP = _config.PROTOCOL_MAP
FLAG_MAP = _config.FLAG_MAP

logger = _logger_mod.get_logger("network_predictor")


class NetworkAnomalyPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.threshold = 0.5
        self._loaded = False

    def load(self):
        self.model = joblib.load(BEST_MODEL_PATH)
        self.scaler = joblib.load(SCALER_PATH)
        self.feature_names = joblib.load(FEATURE_NAMES_PATH)
        with open(MODEL_REPORT_PATH) as f:
            report = json.load(f)
        self.threshold = report.get("optimal_threshold", 0.5)
        self._loaded = True
        logger.info(f"Network predictor loaded | threshold={self.threshold}")
        return self

    def _engineer_features(self, record: dict) -> dict:
        r = record.copy()
        r["protocol_encoded"] = PROTOCOL_MAP.get(r.get("protocol_type", "tcp"), 0)
        r["flag_encoded"] = FLAG_MAP.get(r.get("flag", "SF"), 0)
        r["service_attack_rate"] = r.get("service_attack_rate", 0.465)
        r["service_frequency"] = r.get("service_frequency", 1000)

        src = r.get("src_bytes", 0)
        dst = r.get("dst_bytes", 0)
        r["bytes_ratio"] = src / (dst + 1)
        r["total_bytes"] = src + dst
        r["bytes_log"] = np.log1p(r["total_bytes"])
        r["src_bytes_log"] = np.log1p(src)
        r["dst_bytes_log"] = np.log1p(dst)
        r["is_zero_src"] = int(src == 0)
        r["is_zero_dst"] = int(dst == 0)
        r["high_src_bytes"] = int(src > 10000)

        serror = r.get("serror_rate", 0)
        rerror = r.get("rerror_rate", 0)
        r["error_rate_sum"] = serror + rerror
        r["srv_error_rate_sum"] = r.get("srv_serror_rate", 0) + r.get("srv_rerror_rate", 0)
        r["high_error"] = int(r["error_rate_sum"] > 0.5)
        r["connection_density"] = r.get("count", 1) * r.get("same_srv_rate", 1)
        r["srv_diversity"] = 1 - r.get("same_srv_rate", 1)
        r["host_srv_concentration"] = r.get("dst_host_srv_count", 0) * r.get("dst_host_same_srv_rate", 0)
        r["suspicious_access"] = int(
            r.get("root_shell", 0) > 0 or r.get("su_attempted", 0) > 0 or
            r.get("num_compromised", 0) > 0 or r.get("num_root", 0) > 0
        )
        r["file_activity"] = r.get("num_file_creations", 0) + r.get("num_shells", 0) + r.get("num_access_files", 0)
        r["network_risk_score"] = (
            r["error_rate_sum"] * 0.25 +
            (1 - r.get("same_srv_rate", 1)) * 0.20 +
            r["service_attack_rate"] * 0.30 +
            (1 if r["protocol_encoded"] == 2 else 0) * 0.15 +
            r["suspicious_access"] * 0.10
        )
        return r

    def predict(self, record: dict) -> Dict:
        if not self._loaded:
            self.load()

        engineered = self._engineer_features(record)
        df = pd.DataFrame([engineered])
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0
        df = df[self.feature_names].fillna(0)
        df_scaled = pd.DataFrame(self.scaler.transform(df), columns=self.feature_names)

        attack_score = float(self.model.predict_proba(df_scaled)[:, 1][0])
        is_attack = attack_score >= self.threshold

        threat_level = (
            "CRITICAL" if attack_score >= 0.85 else
            "HIGH" if attack_score >= 0.65 else
            "MEDIUM" if attack_score >= 0.40 else
            "LOW"
        )

        result = {
            "attack_score": round(attack_score, 6),
            "is_attack": bool(is_attack),
            "threat_level": threat_level,
            "threshold_used": self.threshold,
            "action": "BLOCK" if is_attack else "ALLOW",
            "network_risk_score": round(float(engineered["network_risk_score"]), 4),
            "recommendation": (
                "BLOCK IMMEDIATELY — Critical threat detected" if threat_level == "CRITICAL" else
                "BLOCK AND INVESTIGATE" if threat_level == "HIGH" else
                "MONITOR CLOSELY" if threat_level == "MEDIUM" else
                "ALLOW — Normal traffic"
            )
        }
        logger.info(f"Network prediction | score={attack_score:.4f} | threat={threat_level} | action={result['action']}")
        return result