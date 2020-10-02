import torch

import os
os.environ['OMP_NUM_THREADS'] = '4'

import pandas as pd
import numpy as np


from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split

import gc

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
sub = pd.read_csv("../input/sample_submission.csv")

Y = train.label.values
X = train.drop("label", axis = 1).values.reshape((42000, 1, 28, 28)).astype("float32") / 255
X_test = test.values.reshape((28000,1,28,28)).astype("float32") / 255

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 1024, stratify = Y)
x_train, x_test = torch.from_numpy(x_train), torch.from_numpy(x_test)
y_train, y_test = torch.from_numpy(y_train), torch.from_numpy(y_test)

del train, test
gc.collect()

# input = Variable(torch.randn(1, 1, 28, 28))  # 28 by 28 gray scale picture
# cnn_1 = nn.Conv2d(in_channels = 1, out_channels = 10, kernel_size = 3)(input) # [1, 10, 26, 26]
# activation_1 = F.relu(input = cnn_1) # [1, 10, 26, 26]
# max_pool_1 = F.max_pool2d(input = activation_1, kernel_size = 2) # [1, 10, 13, 13]
# cnn_2 = nn.Conv2d(in_channels = 10, out_channels = 20, kernel_size = 3)(max_pool_1) # [1, 20, 11, 11]
# activation_2 = F.relu(input = cnn_2) # [1, 20, 11, 11]
# max_pool_2 = F.max_pool2d(input = activation_2, kernel_size = 2) # [1, 20, 5, 5]
# def num_flat_features(x):
#     size = x.size()[1:]  # all dimensions except the batch dimension
#     num_features = 1
#     for s in size:
#         num_features *= s
#     return num_features
# flatten_length = num_flat_features(max_pool_2) # 500
# flatten = max_pool_2.view(-1, flatten_length) # [1, 500]
# dense_1 = nn.Linear(in_features = flatten_length, out_features = 120)(flatten) # [1, 120]
# activation_3 = F.relu(input = dense_1)
# dense_2 = nn.Linear(in_features = 120, out_features = 84)(activation_3) # [1, 84]
# activation_4 = F.relu(input = dense_2)
# out = nn.Linear(in_features = 84, out_features = 10)(activation_4) # [1, 10]

class Net(nn.Module):
    """ simple pytorch conv net based on tutorial"""
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 10, 3)
        self.conv2 = nn.Conv2d(10, 20, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(20 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
train_data_loader = DataLoader(TensorDataset(x_train, y_train), batch_size = 32)
# valid_data_loader = DataLoader(TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test)), batch_size = 32)

best_valid_loss = 40
best_epoch = 0

for epoch in range(65535):  # loop over the dataset multiple times

    for idx, [x_batch, y_batch] in enumerate(train_data_loader):

        x_var, y_var = Variable(x_batch), Variable(y_batch)
        y_fitted = net(x_var)
        train_loss = criterion(y_fitted, y_var)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

    valid_pred = net(Variable(x_test))
    valid_loss = criterion(valid_pred, Variable(y_test)).data[0]

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        best_epoch = epoch

    print(" === epoch {} --- validation loss {} --- best loss so far {} at epoch {} ".format(
        epoch, valid_loss, best_valid_loss, best_epoch
    ))

    if epoch - best_epoch >= 3:
        print("==== validation loss haven't improve for 3 rounds, early stopping")
        break

print('Finished Training')

test_pred = net(Variable(torch.from_numpy(X_test))).data.numpy().argmax(axis = 1)
sub['Label'] = test_pred
sub.to_csv("pytorch_cnn_starter_sub.csv", index = False)
