import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple
import sys
sys.path.append(str(Path(__file__).resolve().parent))
from config import RAW_DATA_PATH, TARGET_COLUMN
from logger import get_logger

logger = get_logger("data_loader")


class FraudDataLoader:
    def __init__(self):
        self.data_path = RAW_DATA_PATH
        self.df = None

    def load(self) -> pd.DataFrame:
        logger.info(f"Loading data from {self.data_path}")
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        self.df = pd.read_csv(self.data_path)
        logger.info(f"Loaded {len(self.df):,} rows, {self.df.shape[1]} columns")
        return self.df

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Running data validation checks")
        assert TARGET_COLUMN in df.columns, f"Target column '{TARGET_COLUMN}' missing"
        assert df[TARGET_COLUMN].isin([0, 1]).all(), "Target column has values other than 0 and 1"
        assert df.shape[1] >= 30, f"Expected at least 30 columns, got {df.shape[1]}"

        missing = df.isnull().sum().sum()
        if missing > 0:
            logger.warning(f"Found {missing} missing values — will be handled in preprocessing")

        dupes = df.duplicated().sum()
        if dupes > 0:
            logger.info(f"Removing {dupes} duplicate rows")
            df = df.drop_duplicates().reset_index(drop=True)

        neg_amounts = (df["Amount"] < 0).sum()
        if neg_amounts > 0:
            logger.warning(f"Found {neg_amounts} negative amount values")

        fraud_rate = df[TARGET_COLUMN].mean() * 100
        logger.info(f"Fraud rate: {fraud_rate:.4f}% ({df[TARGET_COLUMN].sum()} fraud / {len(df):,} total)")
        return df

    def load_and_validate(self) -> pd.DataFrame:
        df = self.load()
        df = self.validate(df)
        return df