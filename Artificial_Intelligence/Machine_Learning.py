# 1) Design Model (input, output size, forward pass)
# 2) Construct Loss and Optimizer
# 3) Training Loop
#    - forward pass: compute prediction and loss
#    - backward pass: gradients
#    - update weights

import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# Set Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 0) Prepare Data
X_numpy, Y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)

X = torch.from_numpy(X_numpy.astype(np.float32))
Y = torch.from_numpy(Y_numpy.astype(np.float32))
Y = Y.view(Y.shape[0], 1)

N_samples, N_features = X.shape

# 1) Model
Input_size = N_features
Output_size = 1
model = nn.Linear(Input_size, Output_size)

# 2) Loss and Optimizer
Learning_rate = 0.01
Criterion = nn.MSELoss()
Optimizer = torch.optim.SGD(model.parameters(), lr=Learning_rate)

# 3) Training Loop
Num_epoch = 10000
for epoch in range(Num_epoch):
    # Forward Pass and Loss
    Y_predicted = model(X)
    Loss = Criterion(Y_predicted, Y)

    # Backward Pass
    Loss.backward()

    # Update
    Optimizer.step()
    Optimizer.zero_grad()

    if (epoch + 1) % 10 == 0:
        print(f'epoch: {epoch + 1}, loss = {Loss.item():.4}')

# Plot
with torch.no_grad():
    predicted = model(X).detach().numpy()
    plt.plot(X_numpy, Y_numpy, 'ro')
    plt.plot(X_numpy, Y_predicted, 'b')
    plt.show()
