
# dataset_utils.py

from sklearn.preprocessing import StandardScaler


def fit_scaler(X_train):
    """
    Fit scaler using training data only
    """
    scaler = StandardScaler()
    scaler.fit(X_train)

    return scaler


def transform_dataset(X, scaler):
    """
    Apply fitted scaler to dataset
    """
    return scaler.transform(X)