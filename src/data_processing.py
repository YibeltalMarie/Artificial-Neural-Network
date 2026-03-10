
import pandas as pd
import numpy as np


def load_data(file_path):
    """
    Load dataset and remove unused columns
    """
    df = pd.read_csv(file_path)

    df = df.drop(['instant', 'dteday', 'casual', 'registered'], axis=1)

    return df


def split_data(df):
    """
    Shuffle dataset and split into
    60% train, 20% validation, 20% test
    """

    df = df.sample(frac=1, random_state=42)

    train_end = int(0.6 * len(df))
    valid_end = int(0.8 * len(df))

    train = df[:train_end]
    valid = df[train_end:valid_end]
    test = df[valid_end:]

    return train, valid, test


def split_features_target(dataframe):
    """
    Separate features and target
    """
    X = dataframe[dataframe.columns[:-1]].values
    y = dataframe[dataframe.columns[-1]].values

    return X, y