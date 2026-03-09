import torch
import torch.nn as nn

def train_model(model, x_train, y_train, x_valid, y_valid):

    y_train = y_train.view(-1,1)
    y_valid = y_valid.view(-1,1)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 200

    train_losses = []
    valid_losses = []

    for epoch in range(epochs):

        # TRAINING
        model.train()

        y_pred = model(x_train)

        loss = loss_fn(y_pred, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # VALIDATION
        model.eval()

        with torch.no_grad():

            y_val_pred = model(x_valid)

            val_loss = loss_fn(y_val_pred, y_valid)

        train_losses.append(loss.item())
        valid_losses.append(val_loss.item())

        if epoch % 20 == 0:
            print(f"Epoch {epoch} | Train Loss: {loss.item():.4f} | Valid Loss: {val_loss.item():.4f}")

    return train_losses, valid_losses