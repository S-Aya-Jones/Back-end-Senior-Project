# Imports
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import torch.nn.functional as F

# Fuzzy Controller

warmNorm = ctrl.Antecedent(np.arange(0, 1, 0.1), 'Warm Norm')
hotNorm = ctrl.Antecedent(np.arange(0, 1, 0.1), 'Hot Norm')
coolNorm = ctrl.Antecedent(np.arange(0, 1, 0.1), 'Cool Norm')
coldNorm = ctrl.Antecedent(np.arange(0, 1, 0.1), 'Cold Norm')

thermStat = ctrl.Consequent(np.arange(65, 76, 1), 'Set Thermostat')

hotNorm.automf(3)
warmNorm.automf(3)
coolNorm.automf(3)
coldNorm.automf(3)

thermStat['Hot'] = fuzz.trimf(thermStat.universe, [65, 65, 70])
thermStat['Warm'] = fuzz.trimf(thermStat.universe, [65, 68, 72])
thermStat['Cool'] = fuzz.trimf(thermStat.universe, [68, 70, 75])
thermStat['Cold'] = fuzz.trimf(thermStat.universe, [70, 75, 75])
hotNorm.view()
warmNorm.view()
coolNorm.view()
coldNorm.view()
thermStat.view()

rule1 = ctrl.Rule(hotNorm['good'] & warmNorm['average'] & coolNorm['poor'] & coldNorm['poor'], thermStat['Hot'])
rule2 = ctrl.Rule(hotNorm['average'] & warmNorm['good'] & coolNorm['average'] & coldNorm['poor'], thermStat['Warm'])
rule3 = ctrl.Rule(hotNorm['poor'] & warmNorm['average'] & coolNorm['good'] & coldNorm['average'], thermStat['Cool'])
rule4 = ctrl.Rule(hotNorm['poor'] & warmNorm['poor'] & coolNorm['average'] & coldNorm['good'], thermStat['Cold'])
rule5 = ctrl.Rule(hotNorm['poor'] & warmNorm['poor'] & coolNorm['poor'] & coldNorm['poor'], thermStat['Hot'])
rule6 = ctrl.Rule(hotNorm['poor'] & warmNorm['poor'] & coolNorm['poor'] & coldNorm['good'], thermStat['Cold'])
rule7 = ctrl.Rule(hotNorm['poor'] & warmNorm['poor'] & coolNorm['good'] & coldNorm['poor'], thermStat['Cool'])
rule8 = ctrl.Rule(hotNorm['poor'] & warmNorm['good'] & coolNorm['poor'] & coldNorm['poor'], thermStat['Warm'])
rule9 = ctrl.Rule(hotNorm['good'] & warmNorm['poor'] & coolNorm['poor'] & coldNorm['poor'], thermStat['Hot'])

thermStat_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
thermStat_set = ctrl.ControlSystemSimulation(thermStat_ctrl)

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
print(Norms)

Norms_Hot = F.one_hot(Norms)


# Model architecture
Ann_Model = nn.Sequential(
    nn.Linear(2, 64),   # Input layer
    nn.ReLU(),          # Activation
    nn.Linear(64, 64),  # Hidden layer
    nn.ReLU(),          # Activation
    nn.Linear(64, 4),   # Output layer
)

# Loss function
loss_fun = nn.CrossEntropyLoss()

# Optimizer
optimizer = torch.optim.SGD(Ann_Model.parameters(), lr=.01)

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

    # Fuzzy Controller
    for i in range(len(probabilities)):
        thermStat_set.input['Cold Norm'] = probabilities[i][0].detach().numpy()
        thermStat_set.input['Cool Norm'] = probabilities[i][1].detach().numpy()
        thermStat_set.input['Warm Norm'] = probabilities[i][2].detach().numpy()
        thermStat_set.input['Hot Norm'] = probabilities[i][3].detach().numpy()

        thermStat_set.compute()
        predicted_temps.append(thermStat_set.output['Set Thermostat'])
        # print(thermStat_set.output['Set Thermostat'])

    # Compute loss (instead of y_hat it should be outputs from fuzzy controller)
    # a = torch.FloatTensor(predicted_temps)
    print(probabilities)
    loss = loss_fun(probabilities, Norms)
    losses[epoch] = loss

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Compute accuracy
    matches = torch.argmax(probabilities, axis=0) == Norms_Hot  # Booleans (false/true)
    matches_numeric = matches.float()               # Convert to numbers (0/1)
    acc = 100*torch.mean(matches_numeric)           # Average and x100
    accuracy.append(acc)                            # Add to list of accuracies

# Final forward pass
predictions = Ann_Model(data)

pred_labels = torch.argmax(predictions, axis=1)
total_acc = 100*torch.mean((pred_labels == Norms).float())

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

