import pandas as pd
import json
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent))
from config import TRAIN_CLEANED_PATH, TEST_CLEANED_PATH, TRAIN_ENGINEERED_PATH, TEST_ENGINEERED_PATH, META_PATH
from logger import get_logger

logger = get_logger("data_loader")


class NetworkDataLoader:

    def load_meta(self) -> dict:
        with open(META_PATH) as f:
            return json.load(f)

    def load_train(self):
        logger.info("Loading train engineered data")
        df = pd.read_csv(TRAIN_ENGINEERED_PATH)
        labels = pd.read_csv(TRAIN_CLEANED_PATH)["is_attack"]
        logger.info(f"Train: {df.shape} | Attack rate: {labels.mean()*100:.2f}%")
        return df, labels

    def load_test(self):
        logger.info("Loading test engineered data")
        df = pd.read_csv(TEST_ENGINEERED_PATH)
        labels = pd.read_csv(TEST_CLEANED_PATH)["is_attack"]
        logger.info(f"Test: {df.shape} | Attack rate: {labels.mean()*100:.2f}%")
        return df, labels