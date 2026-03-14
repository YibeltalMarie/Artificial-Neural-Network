import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



def show_correlation_matrix(df):
    df = df.drop(['dteday'], axis=1)
    corr = df.corr()

    plt.figure(figsize=(14,10))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix of Bike Sharing Dataset")
    # Get project root directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    results_dir = os.path.join(base_dir, "results")

    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.join(results_dir, "correlation_matrix.png")
    plt.savefig(save_path)
    plt.close()

def load_data(file_path):
    """
    Load dataset and remove unused columns
    """
    df = pd.read_csv(file_path)
    return df

def drop_features(df):
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
    