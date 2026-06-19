import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict
import sys
sys.path.append(str(Path(__file__).resolve().parent))
from config import REPORT_DIR
from logger import get_logger

logger = get_logger("explainer")


class LoanExplainer:
    def __init__(self, model):
        self.model = model
        self.explainer = shap.TreeExplainer(model)
        self.shap_values = None
        self.feature_names = None

    def compute_shap(self, X: pd.DataFrame, sample_size: int = 500):
        X_sample = X.iloc[:sample_size]
        self.feature_names = list(X_sample.columns)
        raw = self.explainer.shap_values(X_sample)
        self.shap_values = raw[1] if isinstance(raw, list) else raw
        logger.info(f"SHAP computed on {len(X_sample)} samples")
        return self.shap_values

    def global_importance(self) -> pd.Series:
        return pd.Series(
            np.abs(self.shap_values).mean(axis=0),
            index=self.feature_names
        ).sort_values(ascending=False)

    def explain_single(self, idx: int) -> Dict:
        shap_vals = self.shap_values[idx]
        contributions = pd.Series(shap_vals, index=self.feature_names)
        return {
            "top_default_signals": contributions.nlargest(5).to_dict(),
            "top_safe_signals": contributions.nsmallest(5).to_dict()
        }

    def plot_summary(self, X: pd.DataFrame):
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            self.shap_values, X.iloc[:len(self.shap_values)],
            max_display=15, show=False
        )
        plt.tight_layout()
        path = REPORT_DIR / "shap_summary.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"SHAP plot saved to {path}")