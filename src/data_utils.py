
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


def to_tensor(X, y):
    """
    Convert numpy arrays to PyTorch tensors
    """
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    return X_tensor, y_tensor