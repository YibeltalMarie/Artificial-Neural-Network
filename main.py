
# main.py
import pandas as pd
from src.data_processing import split_data, split_features_target
from src.data_utils import fit_scaler, transform_dataset, to_tensor
from src.model import RegressionANN
from src.train import train_model
from src.evaluate import evaluate_model
import torch

# Load data
df = pd.read_csv('./data/hour.csv')

# Drop unnecessary columns
df = df.drop(['instant', 'dteday', 'casual', 'registered'], axis=1)

# Split into train, validation, test
train_df, valid_df, test_df = split_data(df)

# Split into features and target
X_train, y_train = split_features_target(train_df)
X_valid, y_valid = split_features_target(valid_df)
X_test, y_test = split_features_target(test_df)


# Fit scaler on training data and transform all sets
scaler = fit_scaler(X_train)
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
plt.show()