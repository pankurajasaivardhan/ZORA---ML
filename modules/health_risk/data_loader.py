import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent))
from config import (
    HEART_RAW_PATH, DIABETES_RAW_PATH,
    HEART_CLEANED_PATH, DIABETES_CLEANED_PATH
)
from logger import get_logger

logger = get_logger("data_loader")


class HealthDataLoader:

    def load_heart(self) -> pd.DataFrame:
        if HEART_CLEANED_PATH.exists():
            logger.info("Loading cleaned heart data")
            return pd.read_csv(HEART_CLEANED_PATH)
        logger.info("Loading raw heart data")
        df = pd.read_csv(HEART_RAW_PATH)
        assert "target" in df.columns
        missing = df.isnull().sum().sum()
        if missing > 0:
            logger.warning(f"Heart data has {missing} missing values")
        df.to_csv(HEART_CLEANED_PATH, index=False)
        logger.info(f"Heart data: {len(df)} patients | Disease rate: {df['target'].mean()*100:.1f}%")
        return df

    def load_diabetes(self) -> pd.DataFrame:
        if DIABETES_CLEANED_PATH.exists():
            logger.info("Loading cleaned diabetes data")
            return pd.read_csv(DIABETES_CLEANED_PATH)
        logger.info("Loading raw diabetes data and cleaning zeros")
        df = pd.read_csv(DIABETES_RAW_PATH)
        zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
        for col in zero_cols:
            df[col] = df[col].replace(0, np.nan)
            df[col] = df[col].fillna(df[col].median())
        df.to_csv(DIABETES_CLEANED_PATH, index=False)
        logger.info(f"Diabetes data: {len(df)} patients | Diabetes rate: {df['Outcome'].mean()*100:.1f}%")
        return df