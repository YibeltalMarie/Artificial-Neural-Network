import torch
import torch.nn as nn

def train_model(model, x_train, y_train, x_valid, y_valid):


    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 500

    train_losses = []
    valid_losses = []

    for epoch in range(epochs):

        # TRAINING
        model.train()

        y_pred = model(x_train)

        loss = loss_fn(y_train, y_pred)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # VALIDATION
        model.eval()

        with torch.no_grad():

            y_val_pred = model(x_valid)

            val_loss = loss_fn(y_valid, y_val_pred)

        train_losses.append(loss.item())
        valid_losses.append(val_loss.item())

        if epoch % 20 == 0:
            # print("Training prediction: ", y_pred)
            # print("Training Actual : ", y_train)
            # print('Valid Prediction: ', y_val_pred)
            # print('Valid Actual: ', y_valid)
            print(f"Epoch {epoch} | Train MSE: {loss.item():.4f} | Valid MSE: {val_loss.item():.4f}")

    return train_losses, valid_losses