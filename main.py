
import pandas as pd
from src.data_preprocessing import show_correlation_matrix, load_data, drop_features, split_data, split_features_target
from src.data_utils import fit_scaler, transform_dataset, to_tensor
from src.model import RegressionANN
from src.train import train_model
from src.evaluate import evaluate_model
import torch

# Load data
df = load_data('./data/hour.csv')
show_correlation_matrix(df)
df = drop_features(df)

# Split into train, validation, test
train_df, valid_df, test_df = split_data(df)

# Split into features and target
X_train, y_train = split_features_target(train_df)
X_valid, y_valid = split_features_target(valid_df)
X_test, y_test = split_features_target(test_df)


y_train = y_train.reshape(-1, 1)
y_valid = y_valid.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)


# Fit scaler on training data and transform all sets
scaler = fit_scaler(X_train)
y_scaler = fit_scaler(y_train)

# Transform all targets/Labels
y_train = transform_dataset(y_train, y_scaler)
y_valid = transform_dataset(y_valid, y_scaler)
y_test = transform_dataset(y_test, y_scaler)

# Transform all features
X_train = transform_dataset(X_train, scaler)
X_valid = transform_dataset(X_valid, scaler)
X_test = transform_dataset(X_test, scaler)

# Convert to PyTorch tensors
X_train, y_train = to_tensor(X_train, y_train)
X_valid, y_valid = to_tensor(X_valid, y_valid)
X_test, y_test = to_tensor(X_test, y_test)

# Initialize model
input_dim = X_train.shape[1]
model = RegressionANN(input_dim)

# Train the model
train_losses, valid_losses = train_model(model, X_train, y_train, X_valid, y_valid)

# Evaluate on test data
evaluate_model(model, X_test, y_test)

# Optional: Plot losses
import matplotlib.pyplot as plt

plt.plot(train_losses, label='Train Loss')
plt.plot(valid_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training & Validation Loss')
plt.legend()
plt.savefig("./results/loss.png")