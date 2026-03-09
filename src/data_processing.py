
# data_preprocessing.py

import pandas as pd
import numpy as np


def load_data(file_path):
    """
    Load dataset and drop unnecessary columns
    """
    df = pd.read_csv(file_path)

    df = df.drop(['instant', 'dteday', 'casual', 'registered'], axis=1)

    return df


def split_data(df):
    """
    Shuffle and split dataset into
    train (60%), validation (20%), test (20%)
    """

    train, valid, test = np.split(
        df.sample(frac=1, random_state=42),
        [int(0.6 * len(df)), int(0.8 * len(df))]
    )

    return train, valid, test