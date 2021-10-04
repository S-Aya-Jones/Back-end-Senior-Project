# Imports
from Artificial_Intelligence.FuzzyController import FuzzyController
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F


# Set Device #
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Import Data #
avgTemp = pd.read_csv("./data/AvgTempCSV.csv")
maxTemp = pd.read_csv("./data/MaxTempCSV.csv")
minTemp = pd.read_csv("./data/MinTempCSV.csv")

df = pd.DataFrame(avgTemp, columns=['Date'])
df['Avg Temp'] = (avgTemp['Value'])
df['Max Temp'] = (maxTemp['Value'])
df['Thermostat'] = (avgTemp['Thermostat'])
df['TempNorm'] = (avgTemp['TempNorm'])
df['Min Temp'] = (minTemp['Value'])

# convert from pandas dataframe to tensor
data = torch.tensor(df[df.columns[1:3]].values).float()
labels = torch.tensor(df[df.columns[3:4]].values).long()

Norms = torch.zeros(len(data), dtype=torch.long)
Norms[df.TempNorm == 'Cool'] = 1
Norms[df.TempNorm == 'Warm'] = 2
Norms[df.TempNorm == 'Hot'] = 3

Norms_Hot = F.one_hot(Norms)

# Model architecture
Ann_Model = nn.Sequential(
    nn.Linear(2, 64),  # Input layer
    nn.ReLU(),  # Activation
    nn.Linear(64, 64),  # Hidden layer
    nn.ReLU(),  # Activation
    nn.Linear(64, 64),  # Hidden layer
    nn.ReLU(),  # Activation
    nn.Linear(64, 64),  # Hidden layer
    nn.ReLU(),  # Activation
    nn.Linear(64, 4),  # Output layer
)

# Loss function
loss_fun = nn.CrossEntropyLoss()

# Optimizer
optimizer = torch.optim.SGD(Ann_Model.parameters(), lr=.001)

# epochs
num_epochs = 1000

# initialize losses
losses = torch.zeros(num_epochs)
accuracy = []
predicted_temps = []

# Training loop
for epoch in range(num_epochs):
    # Forward pass
    y_hat = Ann_Model(data)

    sm = nn.Softmax(dim=1)
    probabilities = sm(y_hat)

    # Call Fuzzy Controller
    a = FuzzyController(probabilities)
    predicted_temps.append(a.inference())

    # Compute loss (instead of y_hat it should be outputs from fuzzy controller)
    # a = torch.FloatTensor(predicted_temps)
    loss = loss_fun(probabilities, Norms)
    losses[epoch] = loss

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Compute accuracy
    matches = torch.argmax(probabilities, axis=1) == Norms  # Booleans (false/true)
    matches_numeric = matches.float()  # Convert to numbers (0/1)
    acc = 100 * torch.mean(matches_numeric)  # Average and x100
    accuracy.append(acc)  # Add to list of accuracies

# Final forward pass
predictions = Ann_Model(data)
print(predictions)
pred_labels = torch.argmax(predictions, axis=1)
print(pred_labels)
total_acc = 100 * torch.mean((pred_labels == Norms).float())

# Report accuracy
print('Final accuracy: %g%%' % total_acc)

fig, ax = plt.subplots(1, 2, figsize=(13, 4))

ax[0].plot(losses.detach())
ax[0].set_ylabel('Loss')
ax[0].set_xlabel('Epoch')
ax[0].set_title('Losses')

ax[1].plot(accuracy)
ax[1].set_ylabel('Accuracy')
ax[1].set_xlabel('Epoch')
ax[1].set_title('Accuracy')
plt.show()

# sm = nn.Softmax(1)
# torch.sum(y_hat, axis=1)
#
# fig = plt.figure(figsize=(10, 4))
#
# plt.plot(sm(y_hat.detach()), 's-', markerfacecolor='w')
# plt.xlabel('Stimulus number')
# plt.ylabel('Probability')
# plt.legend(['Cold', 'Cool', 'Warm', 'Hot'])
# plt.show()
