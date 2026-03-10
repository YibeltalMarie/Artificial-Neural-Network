
import torch
import torch.nn as nn
import numpy as np


def evaluate_model(model, x_test, y_test):

    model.eval()

    # reshape targets
    y_test = y_test.view(-1, 1)

    with torch.no_grad():

        predictions = model(x_test)

        mse_loss = nn.MSELoss()
        mse = mse_loss(predictions, y_test).item()

        rmse = np.sqrt(mse)

        mae = torch.mean(torch.abs(predictions - y_test)).item()

    print("Evaluation Results")
    print("------------------")
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAE  : {mae:.4f}")

    return mse, rmse, mae