import os
import torch
import torch.nn as nn
import numpy as np
from src.data_utils import fit_scaler, transform_dataset, to_tensor
import matplotlib.pyplot as plt

def evaluate_model(model, x_test, y_test):

    model.eval()

    # reshape targets
    # y_test = y_test.view(-1, 1)
    # y_scaler = fit_scaler(y_test)
    # y_test = transform_dataset(y_test, y_scaler)
    with torch.no_grad():

        predictions = model(x_test)

    
        mse_loss = nn.MSELoss()
        mse = mse_loss(y_test, predictions)

        rmse = np.sqrt(mse)

        mae = torch.mean(torch.abs(predictions - y_test)).item()
        # MAPE 
        epsilon = 1e-7  # to avoid division by zero
        mape = torch.mean(torch.abs((y_test - predictions) / (y_test + epsilon))).item()
    


    print("Evaluation Results")
    print("------------------")
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAE  : {mae:.4f}")
    print(f"MAPE: {mape:.2f}")

    return mse, rmse, mae, mape


def plot_metrics(mse, rmse, mae, mape, results_dir):
    save_path = os.path.join(results_dir, "metric.png")
    metrics = ["MSE", "RMSE", "MAE", "MAPE"]
    values = [mse, rmse, mae, mape]

    plt.figure(figsize=(8,5))

    plt.bar(metrics, values)

    plt.title("Model Error Metrics")
    plt.xlabel("Metrics")
    plt.ylabel("Error Value")

    plt.savefig(save_path)
    plt.close()