# -*- coding: utf-8 -*-
#%%
import sys
sys.path.append("/usr/local/lib/python3.7/dist-packages") #Path to python on Linux

#%%
import pandas as pd
import numpy as np

#%%
"""
Loading data into memory and making an attempt to extract 'measurable' parameters.
"""
data = pd.read_excel("../input/covid19/dataset.xlsx") #Needs xlrd package
X = data._get_numeric_data()
X = X.loc[:, ~X.columns.isin(X.columns[1:4])] #Removing one-hot-encoded parameters as they aren't measurable
X = X.loc[:, X.isnull().sum()/X.shape[0] < 0.9] #Removing overly sparse data
"""
Setting NaNs to 0. As this may cause bias in the network, I'll add at least two layers,
one to prepare the data and another for the actual approximation. This way, the network should
learn to not use zero values as they appear.
"""
X = np.nan_to_num(X.to_numpy())
Y = data['SARS-Cov-2 exam result']
Y = np.array([0 if u=="negative" else 1 for u in Y])


#%%
"""
Perceptron (Uses PyTorch)
"""
import torch
#Will use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
Splitting data into train and test subsets. I will use 80%-20% respectively.
Reference: https://pytorch.org/tutorials/beginner/pytorch_with_examples.html
"""

X = torch.from_numpy(X).to(device)
Y = torch.from_numpy(Y).to(device)

train_size = int(0.8 * len(X))
test_size = len(X) - train_size
x, x_test = torch.utils.data.random_split(X, [train_size, test_size])
y, y_test = torch.utils.data.random_split(Y, [train_size, test_size])

x,x_test = x.dataset[x.indices].float(), x_test.dataset[x_test.indices].float()
y,y_test = y.dataset[y.indices].float(), y_test.dataset[y_test.indices].float()

"""
Building network
"""
# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = X.shape[0], X.shape[1], X.shape[1], 1

perceptron = torch.nn.Sequential(
    torch.nn.Sigmoid(), #This layer normalizes everyone to be in (0,1)
    torch.nn.Linear(D_in, H).to(device), #First layer
    torch.nn.Sigmoid(),
    torch.nn.Linear(H,H).to(device), #Second layer
    torch.nn.Sigmoid(),
    torch.nn.Linear(H, D_out).to(device), #Output layer
)
loss_fn = torch.nn.MSELoss(reduction='sum')

learning_rate = 1e-6
optimizer = torch.optim.SGD(perceptron.parameters(), lr=learning_rate)

"""
Training network
"""

for t in range(500):
    # Forward pass: compute predicted y by passing x to the perceptron.
    y_pred = perceptron(x)

    # Compute and print loss.
    loss = loss_fn(y_pred.view(-1), y)
    if t % 100 == 99:
        print(t, loss.item())

    # Before the backward pass, use the optimizer object to zero all of the
    # gradients for the variables it will update (which are the learnable
    # weights of the perceptron). This is because by default, gradients are
    # accumulated in buffers( i.e, not overwritten) whenever .backward()
    # is called. Checkout docs of torch.autograd.backward for more details.
    optimizer.zero_grad()

    # Backward pass: compute gradient of the loss with respect to perceptron
    # parameters
    loss.backward()

    # Calling the step function on an Optimizer makes an update to its
    # parameters
    optimizer.step()

"""
Testing
"""
predictions = perceptron(x_test)
epsilon = 0.5 #Middle of the sigmoid
predictions = [0 if u<epsilon else 1 for u in predictions]
accuracy = np.array([1 if y_t==y_p else 0 for y_t,y_p in zip(y_test, predictions)]) #Counting matches
print("accuracy: ", sum(accuracy)/len(accuracy))

"""
Smallest accuracy ouf of ten: 0.895
Highest accuracy out of ten: 0.923
"""
