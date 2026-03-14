
import torch
import torch.nn as nn
import numpy as np
from src.data_utils import fit_scaler, transform_dataset, to_tensor


def evaluate_model(model, x_test, y_test):

    model.eval()

    # reshape targets
    # y_test = y_test.view(-1, 1)
    # y_scaler = fit_scaler(y_test)
    # y_test = transform_dataset(y_test, y_scaler)
    with torch.no_grad():

        predictions = model(x_test)

        # # convert to numpy
        # predictions = predictions.detach().numpy()

        # # convert back to original bike counts
        # predictions = y_scaler.inverse_transform(predictions)

        # # same for true values
        # y_test = y_scaler.inverse_transform(y_test)

        # y_test = torch.tensor(y_test, dtype=torch.float32)
        # pred = torch.tensor(pred, dtype=torch.float32)
        mse_loss = nn.MSELoss()
        mse = mse_loss(y_test, predictions)

        rmse = np.sqrt(mse)

        # mae = torch.mean(torch.abs(predictions - y_test)).item()

    print("Evaluation Results")
    print("------------------")
    # print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")
    # print(f"MAE  : {mae:.4f}")

    return mse