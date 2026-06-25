import pandas as pd
import json
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent))
from config import RAW_DATA_PATH, CLEANED_DATA_PATH
from logger import get_logger

logger = get_logger("data_loader")


class ChurnDataLoader:

    def load_and_clean(self) -> pd.DataFrame:
        if CLEANED_DATA_PATH.exists():
            logger.info("Loading cleaned churn data")
            return pd.read_csv(CLEANED_DATA_PATH)

        logger.info("Loading raw churn data")
        df = pd.read_csv(RAW_DATA_PATH)
        df["Churn_Binary"] = (df["Churn"] == "Yes").astype(int)
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

        df.to_csv(CLEANED_DATA_PATH, index=False)
        logger.info(f"Cleaned data: {len(df):,} customers | Churn rate: {df['Churn_Binary'].mean()*100:.2f}%")
        return df