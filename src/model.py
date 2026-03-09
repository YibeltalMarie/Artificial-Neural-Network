
import torch.nn as nn

class RegressionANN(nn.Module):
  def __init__(self, input_dim):
    super(RegressionANN,self).__init__()

    #Hidden Layers (fully connected)
    self.hidden1 = nn.Linear(input_dim, 64)
    self.hidden2 = nn.Linear(64, 32)
    self.hidden3 = nn.Linear(32, 16)

    #Output Layer
    self.output = nn.Linear(16, 1)

    #Activation function
    self.relu = nn.ReLU()

  def forward(self, x):
    x = self.relu(self.hidden1(x))
    x = self.relu(self.hidden2(x))
    x = self.relu(self.hidden3(x))
    x = self.output(x)

    return x
