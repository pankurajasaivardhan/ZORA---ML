import pandas as pd
import json
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent))
from config import TRAIN_ENGINEERED_PATH, TEST_ENGINEERED_PATH, META_PATH
from logger import get_logger

logger = get_logger("data_loader")


class EquipmentDataLoader:

    def load_meta(self) -> dict:
        with open(META_PATH) as f:
            return json.load(f)

    def load_train(self) -> pd.DataFrame:
        logger.info("Loading train engineered data")
        df = pd.read_csv(TRAIN_ENGINEERED_PATH)
        logger.info(f"Train: {df.shape}")
        return df

    def load_test(self) -> pd.DataFrame:
        logger.info("Loading test engineered data")
        df = pd.read_csv(TEST_ENGINEERED_PATH)
        logger.info(f"Test: {df.shape}")
        return df