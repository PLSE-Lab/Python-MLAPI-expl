import numpy as np
import pandas as pd
import sklearn as skl

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset
import torchvision.datasets


BATCH_SIZE = None

class MNIST(Dataset):
    def __init__(self, filename, train=True, shuffle=True):
        data = pd.read_csv(filename)

        if shuffle:
            data = skl.utils.shuffle(data)
            data.reset_index(inplace=True, drop=True)

        if train:
            self.images = data.iloc[:, 1:] / 255
            self.labels = data.iloc[:, 0]
        else:
            self.images = data / 255
            self.labels = np.empty(len(data))
            
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if not isinstance(idx, list):
            idx = list(idx)

        images = torch.from_numpy(self.images.iloc[idx].values).float()
        
        labels = torch.from_numpy(np.array(self.labels[idx]))
        
        return images, labels

    def __iter__(self):
        for i in range(0, len(self), BATCH_SIZE):
            yield self[i + np.arange(BATCH_SIZE)]


class NN(nn.Module):
    def __init__(self, incoming, hidden1, hidden2, out, pool_size=4, pool_stride=2):
        super(NN, self).__init__()
        
        ## Assumes incoming is flattened image size
        
        self.pool_size = pool_size
        self.pool_stride = pool_stride
    
        self.l1 = nn.Linear(incoming // pool_stride - int(incoming**.5), hidden1)
        self.l2 = nn.Linear(hidden1, hidden2)
        self.l3 = nn.Linear(hidden2, out)

    def forward(self, x):
        x = x.view(BATCH_SIZE, 28, 28)

        x = F.max_pool1d(x, self.pool_size, self.pool_stride)

        x = x.view(BATCH_SIZE, self.l1.in_features)

        x = F.relu(self.l1(x))
        
        x = F.relu(self.l2(x))

        x = self.l3(x)

        return x


## Read dataset
BATCH_SIZE = 4
train_set = MNIST('../input/train.csv')

## Initialize
network = NN(784, 16, 16, 10)

criterion = nn.CrossEntropyLoss()
optimizer = Adam(network.parameters(), lr=0.0001)

## Train
for epoch in range(10):

    total_loss = 0
    for i, data in enumerate(train_set):
        optimizer.zero_grad()
        
        inputs, labels = data

        optimizer.zero_grad()

        outputs = network(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if i % 2000 == 1999:
            print(f'{epoch+1:2.0f}, {i+1:5.0f} - loss: {total_loss / 2000:.3f}')
            total_loss = 0.0

## Test
BATCH_SIZE =1
test_set = MNIST('../input/test.csv', train=False, shuffle=False)

predictions = [int(torch.max(network(inputs).data, 1)[1]) for inputs, labels in test_set]

output = pd.DataFrame()
output['ImageId'] = [i+1 for i in range(len(predictions))]
output['Label'] = predictions

print(output)

output.to_csv('output.csv', index=False)
