#!/usr/bin/env python
import collections
import numpy as np
import sys
import copy
import os
import math

TRACE_PATH = "/root/parallelism_workspace/traces/origin/wdev.total.csv.orig"
PAGE_SIZE = 8

traing_data = []
last_issue = -1
f = open(TRACE_PATH, 'r')
for line in f:
    trace = line.split(" ")
    address = int(trace[2])
    req_size = int(trace[3])

    address = math.floor(address / PAGE_SIZE) * PAGE_SIZE
    address_end = math.ceil((address + req_size) / PAGE_SIZE) * PAGE_SIZE
    req_size = address_end - address
    
    issued = float(trace[0])
    if last_issue != -1:
        traing_data.append([address, issued - last_issue])

    last_issue = issued


# use 2 layer MLP to predict the interval of the next request
# input: address
# output: interval
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Assuming traing_data is a list of [address, interval]
traing_data = np.array(traing_data)

# Normalize data
scaler1 = StandardScaler()
X = scaler1.fit_transform(traing_data[:, 0].reshape(-1, 1))
scaler2 = StandardScaler()
y = scaler2.fit_transform(traing_data[:, 1].reshape(-1, 1))

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to torch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Define the model
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.layers(x)

model = SimpleMLP()

# Define loss function and optimizer
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):  # number of epochs
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = loss_function(outputs, y_train)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# Evaluate the model
model.eval()
with torch.no_grad():
    predicted = model(X_test)
    mse = loss_function(predicted, y_test)
    print(f"Test MSE: {mse.item()}")

import struct
f = open("mlp_model.bin", "wb")
# Save the model
output = model(torch.tensor(X, dtype=torch.float32)).detach().numpy()
input = scaler1.inverse_transform(X)
output = scaler2.inverse_transform(output)
f.write(struct.pack("Q", len(input)))
for i in range(len(input)):
    f.write(struct.pack("Q", int(input[i][0])))
    f.write(struct.pack("Q", int(float(output[i][0]) * 1000000000000)))
