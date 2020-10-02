#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


#load packages: numpy, matplotlib, and pytorch
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
import numpy as np
import pandas as pd
import csv


# In[ ]:


# Put some augmentation on training data
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    #transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),
    transforms.ToTensor()
])

# Test data without augmentation
test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor()
])


# In[ ]:


train_data = pd.read_csv("../input/Kannada-MNIST/train.csv")
dt = train_data.values
x_data = torch.from_numpy(dt[:,1:])
y_data = torch.from_numpy(dt[:,0])

x_cnn_data = x_data.reshape(60000, 1, 28, 28).float()
for data in x_cnn_data:
    data = train_transform(data)
    
train_dataset = TensorDataset(x_cnn_data, y_data)
train_dataset, dev_dataset = random_split(train_dataset, [50000, 10000])


# In[ ]:


test_data = pd.read_csv("../input/Kannada-MNIST/test.csv")
dt_test = test_data.values
x_test_data = torch.from_numpy(dt_test[:,1:]).reshape(-1,1,28,28).float()
for data in x_test_data:
    data = test_transform(data)


# In[ ]:


#set parameters
num_epochs = 20
batch_size = 128
learning_rate = 0.001


# In[ ]:


# Create data generators - they will produce batches
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
dev_loader = DataLoader(dataset=dev_dataset, batch_size=batch_size, shuffle=False)


# In[ ]:


class QzCNN(nn.Module):
    def __init__(self):
        super(QzCNN, self).__init__()
        self.num_channels = 1
        self.image_size = 28
        self.num_labels = 10
        self.conv2d_1 = nn.Conv2d(1, 32, 3, 1, padding=1)
        self.bn1_1 = nn.BatchNorm2d(32)
        self.conv2d_2 = nn.Conv2d(32, 32, 3, 1, padding=1)
        self.bn1_2 = nn.BatchNorm2d(32)
        self.conv2d_3 = nn.Conv2d(32, 64, 3, 1, padding=1)
        self.bn1_3 = nn.BatchNorm2d(64)
        self.conv2d_4 = nn.Conv2d(64, 64, 3, 1, padding=1)
        self.bn1_4 = nn.BatchNorm2d(64)
        self.conv2d_5 = nn.Conv2d(64, 128, 3, 1)
        self.bn1_5 = nn.BatchNorm2d(128)
        self.conv2d_6 = nn.Conv2d(128, 128, 3, 1, padding=1)
        self.bn1_6 = nn.BatchNorm2d(128)
        self.conv2d_7 = nn.Conv2d(128, 256, 3, 1)
        self.bn1_7 = nn.BatchNorm2d(256)
        self.conv2d_8 = nn.Conv2d(256, 256, 3, 1, padding=1)
        self.bn1_8 = nn.BatchNorm2d(256)
    
        self.dense_1 = nn.Linear(4 * 4 * 256, 200)
        self.bn2_1 = nn.BatchNorm1d(200)
        self.dense_2 = nn.Linear(200, 200)
        self.bn2_2 = nn.BatchNorm1d(200)
        #self.Dropout = nn.Dropout(0.5)
        self.dense_3 = nn.Linear(200, 10)


    def forward(self, x):
        x = F.relu(self.bn1_1(self.conv2d_1(x)))
        x = F.relu(self.bn1_2(self.conv2d_2(x)))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.bn1_3(self.conv2d_3(x)))
        x = F.relu(self.bn1_4(self.conv2d_4(x)))
        x = F.relu(self.bn1_5(self.conv2d_5(x)))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.bn1_6(self.conv2d_6(x)))
        x = F.relu(self.bn1_7(self.conv2d_7(x)))
        x = F.relu(self.bn1_8(self.conv2d_8(x)))
        #x = x.permute((0, 2, 3, 1))

        x = x.reshape(-1, 4 * 4 * 256)
        x = F.relu(self.bn2_1(self.dense_1(x)))
        x = F.relu(self.bn2_2(self.dense_2(x)))
        #x = self.Dropout(x)
        x = self.dense_3(x)
        return x


# In[ ]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[ ]:


model = QzCNN().to(device)

lossFunction = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = lr_scheduler.StepLR(optimizer,step_size=5,gamma = 0.7)


# In[ ]:


print(model)


# In[ ]:


def train(model, device, iterator, optimizer, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for (x, y) in iterator:
        
        x = x.to(device).float()
        y = y.to(device)
        
        optimizer.zero_grad()
                
        fx = model(x)
        
        loss = criterion(fx, y)
        
        acc = calculate_accuracy(fx, y)
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


# In[ ]:


def evaluate(model, device, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
        for (x, y) in iterator:

            x = x.to(device).float()
            y = y.to(device)

            fx = model(x)

            loss = criterion(fx, y)

            acc = calculate_accuracy(fx, y)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


# In[ ]:


def calculate_accuracy(fx, y):
    preds = fx.max(1, keepdim=True)[1]
    correct = preds.eq(y.view_as(preds)).sum()
    acc = correct.float()/preds.shape[0]
    return acc


# In[ ]:


def predict(model, test_data):
    predict_list = []
    i = 0
    with torch.no_grad():
        for image in test_data:
            model = model.to(device)
            image = image.to(device).float()
            image = image.unsqueeze(0)
            output = model(image)
            _, predicted = torch.max(output, 1)
            predict_list.append(int(predicted[0]))
            i += 1
            
    return predict_list


# In[ ]:


SAVE_DIR = '../models'
MODEL_SAVE_PATH = os.path.join(SAVE_DIR, 'mnist.pt')

best_valid_loss = float('inf')

if not os.path.isdir(f'{SAVE_DIR}'):
    os.makedirs(f'{SAVE_DIR}')
    
#train network
for epoch in range(num_epochs):
    train_loss, train_acc = train(model, device, train_loader, optimizer, lossFunction)
    valid_loss, valid_acc = evaluate(model, device, dev_loader, lossFunction) 
    
    '''if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), MODEL_SAVE_PATH)'''
    
    #print(f'| Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:05.2f}% | ')
    print(f'| Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:05.2f}% | Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc*100:05.2f}% |')


# In[ ]:


predicted_labels = predict(model, x_test_data)
test = pd.read_csv('../input/Kannada-MNIST/test.csv')
my_submission = pd.DataFrame({'id': test.id, 'Label': predicted_labels})
my_submission.to_csv('submission.csv', index=False)

