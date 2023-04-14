import os
from typing import Tuple

import pandas as pd

from .constants import PATH_DATA


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and test data."""

    print(f"Loading data...")

    df_train = pd.read_csv(os.path.join(PATH_DATA, "train.csv"))
    df_test = pd.read_csv(os.path.join(PATH_DATA, "test.csv"))

    print(f"Data have been loaded!\n"
          f"Train shape: {df_train.shape}\n"
          f"Test shape: {df_test.shape}")

    return df_train, df_test


def split_features_target(df: pd.DataFrame, target_col: str = "site_eui") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Splits dataset on features matrix and targets vector."""

    print(f"Splitting data on features and target column '{target_col}'...")

    df = df.copy()

    features = df.drop(target_col, axis=1)
    target = df[[target_col]]

    print(f"Data have been split!")

    return features, target
