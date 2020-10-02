# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

train = pd.read_csv('../input/train.csv', dtype=np.float32)
test = pd.read_csv('../input/test.csv', dtype=np.float32)

test_numpy = test.values / 255
test = test_numpy.reshape(-1, 1, 28, 28)
test_data = torch.from_numpy(test)

targets_numpy = train.label.values
features_numpy = train.loc[:, train.columns != 'label'].values / 255
features_numpy = features_numpy.reshape(-1, 1, 28, 28)

features_train, features_test, targets_train, targets_test = train_test_split(features_numpy, targets_numpy,
                                                                              test_size=0.2, random_state=42)
featurestrain = torch.from_numpy(features_train)
targetstrain = torch.from_numpy(targets_train).type(torch.LongTensor)

featurestest = torch.from_numpy(features_test)
targetstest = torch.from_numpy(targets_test).type(torch.LongTensor)

batch_size = 100
n_iters = 10000
num_epochs = 5

train = torch.utils.data.TensorDataset(featurestrain, targetstrain)
test = torch.utils.data.TensorDataset(featurestest, targetstest)

train_loader = torch.utils.data.DataLoader(dataset=train, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test, batch_size=batch_size, shuffle=True)


class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=0),
            nn.ReLU(True),
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=0),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout2d(p=0.25),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout2d(p=0.25),

        )

        self.fc = nn.Sequential(
            nn.Linear(64 * 3 * 3, 256),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(256, 10),

        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def train(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    running_loss = 0
    total_steps = len(train_loader)

    for epoch in range(num_epochs):
        for batch_id, (images, labels) in enumerate(train_loader):
            images = Variable(images)
            labels = Variable(labels)

            optimizer.zero_grad()
            outputs = model.forward(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if (batch_id + 1) % 112 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss:{:.4f}'.format(epoch+1, num_epochs,
                                                                        batch_id+1, total_steps, loss.item()))


def validate(model, test_loader):
    model.eval()

    with torch.no_grad():
        correct = 0
        total = 0

        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the test images: {} %'.format(100 * correct / total))


def test(model, test_data):
    model.eval()

    with torch.no_grad():
        outputs = model(test_data)
        _, predicted = torch.max(outputs.data, 1)
        predict_result = predicted.tolist()
        
        predict_result = pd.DataFrame(predict_result)
        inx = np.arange(28000) + 1
        inx = pd.DataFrame(inx)
        result = pd.concat([inx, predict_result], axis=1)
        result.columns = ['ImageId', 'Label']
        result.to_csv('submissions.csv', index=False)


model = MyCNN()
criterion = nn.CrossEntropyLoss()
lr_rate = 0.001
optimizer = optim.Adam(model.parameters(), lr=lr_rate)

if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()

train(model, train_loader, criterion, optimizer, num_epochs)
validate(model, test_loader)
test(model, test_data)