# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F


# Model architecture
class Ann_Model(nn.Module):
    def __init__(self):  # Constructor (initializing model)
        super(Ann_Model, self).__init__()

        # Neural Network layers
        self.input = nn.Linear(2, 15)
        self.hidden1 = nn.Linear(15, 20)
        self.hidden2 = nn.Linear(20, 64)
        self.output = nn.Linear(64, 4)

        # Forward pass

    def forward(self, x_data):
        # Input layer
        x = self.input(x_data)  # Input data
        x = F.relu(x)  # Normalization

        # Hidden layer 1
        x = self.hidden1(x)
        x = F.relu(x)  # Normalization

        # Hidden layer 2
        x = self.hidden2(x)
        x = F.relu(x)  # Normalization

        # Output layer
        x = self.output(x)
        y = torch.softmax(x, dim=1)  # Normalization (probabilities)

        return y
