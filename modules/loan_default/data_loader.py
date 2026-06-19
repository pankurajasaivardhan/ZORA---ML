import pandas as pd
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent))
from config import RAW_DATA_PATH, CLEANED_DATA_PATH, KEY_FEATURES, DEFAULT_STATUSES, TARGET_COLUMN
from logger import get_logger

logger = get_logger("data_loader")


class LoanDataLoader:
    def __init__(self):
        self.df = None

    def load_raw(self) -> pd.DataFrame:
        logger.info(f"Loading raw data from {RAW_DATA_PATH}")
        if not RAW_DATA_PATH.exists():
            raise FileNotFoundError(f"Raw data not found: {RAW_DATA_PATH}")
        df = pd.read_csv(RAW_DATA_PATH, low_memory=False)
        logger.info(f"Loaded {len(df):,} rows, {df.shape[1]} columns")
        return df

    def load_cleaned(self) -> pd.DataFrame:
        logger.info(f"Loading cleaned data from {CLEANED_DATA_PATH}")
        if not CLEANED_DATA_PATH.exists():
            raise FileNotFoundError(f"Cleaned data not found: {CLEANED_DATA_PATH}")
        df = pd.read_csv(CLEANED_DATA_PATH)
        logger.info(f"Loaded {len(df):,} rows")
        return df

    def create_target(self, df: pd.DataFrame) -> pd.DataFrame:
        df[TARGET_COLUMN] = df["loan_status"].isin(DEFAULT_STATUSES).astype(int)
        return df

    def select_and_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        available = [c for c in KEY_FEATURES if c in df.columns]
        df = df[available].copy()
        df["emp_length"] = (
            df["emp_length"].str.replace(" years", "").str.replace(" year", "")
            .str.replace("+", "").str.replace("< 1", "0")
        )
        df["emp_length"] = pd.to_numeric(df["emp_length"], errors="coerce")
        df["revol_util"] = pd.to_numeric(df["revol_util"], errors="coerce")
        df = df.dropna(subset=["annual_inc", "dti", "revol_util", "fico_range_low"])
        logger.info(f"After cleaning: {len(df):,} rows | Defaults: {df[TARGET_COLUMN].sum():,}")
        return df

    def load_and_prepare(self) -> pd.DataFrame:
        if CLEANED_DATA_PATH.exists():
            logger.info("Cleaned data found — loading directly")
            return self.load_cleaned()
        df = self.load_raw()
        df = self.create_target(df)
        df = self.select_and_clean(df)
        df.to_csv(CLEANED_DATA_PATH, index=False)
        logger.info(f"Cleaned data saved to {CLEANED_DATA_PATH}")
        return df