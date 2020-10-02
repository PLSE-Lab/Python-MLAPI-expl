#!/usr/bin/env python
# coding: utf-8

# In[10]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[11]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision import datasets
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image
from sklearn.metrics import classification_report
from IPython.display import display
from torch.utils.data import Dataset


# In[13]:


train_data = "../input/fruits-360_dataset/fruits-360/Training/"
test_data = "../input/fruits-360_dataset/fruits-360/Test/"


# In[14]:


_mean = [0.485, 0.456, 0.406]
_std = [0.229, 0.224, 0.225]

epoches = 2


# In[15]:


class MyDataset(Dataset):
    def __init__(self, root, transform = None):
        super().__init__()
        self.root = root
        self.image_arr = os.listdir(root)
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(_mean, _std),])
        self.images = []
        
        for n, i in enumerate(self.image_arr):
            i += '/'
            self.images.extend([(i + j, n) for j in os.listdir(self.root + i)])
         
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):

        image = Image.open(self.root + self.images[index][0])
        if(self.transform):
            return (self.transform(image), self.images[index][1])
        return image
   


# In[16]:


train_dataset = MyDataset(train_data)
test_dataset = MyDataset(test_data)

train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=50, shuffle=False)  

val_ds = []
test = []


# In[17]:


for i in range(len(test_dataset)):
    if (i%2 == 0):
        val_ds.append(test_dataset[i])
    else:
        test.append(test_dataset[i])


# In[18]:


val_dl = DataLoader(val_ds, batch_size =100, shuffle=False)
test_dl = DataLoader(test, batch_size =100, shuffle=False)


# In[19]:


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=3)      
        self.conv3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size=5)
        self.pool4 = nn.MaxPool2d(kernel_size=4)
        
        self.fc5 = nn.Linear(in_features = 7 * 7 * 64,out_features = 103)
       
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.pool2(x))
        x = self.conv3(x)
        x = F.relu(self.pool4(x))
        x = x.view(-1, 7*7*64)
        x = self.fc5(x)
        return x


# In[20]:


model = ConvNet().cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)


# In[28]:


model.train()

for epoch in range(2):
    running_loss = 0.0
    i = 0
    for xx, yy in train_loader:
        xx, yy = xx.cuda(), yy.cuda()
        optimizer.zero_grad()
        outputs = model(xx)
        loss = criterion(outputs, yy)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        i += 1
        if i % 100 == 0:
            print(i, len(train_loader))
            
    model.eval()
    y_test = []
    y_pred = []

    with torch.no_grad():
        correct = 0
        total = 0
        i = 0
        for xx, yy in val_dl:
            xx, yy = xx.cuda(), yy.cuda()
            images = xx
            labels = yy
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            y_pred.extend(predicted.tolist())
            y_test.extend(yy.tolist())
            i += 1
            if (i%100 == 0):
                print (i, len(val_loader))
        
    print(classification_report(y_test, y_pred)) 


# In[29]:


model.eval()
y_test = []
y_pred = []

with torch.no_grad():
    correct = 0
    total = 0
    i = 0
    for xx, yy in test_dl:
        xx, yy = xx.cuda(), yy.cuda()
        images = xx
        labels = yy
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        y_pred.extend(predicted.tolist())
        y_test.extend(yy.tolist())
        i += 1
        if (i%100 == 0):
            print (i, len(test_loader))
        
    print(classification_report(y_test, y_pred)) 


# In[30]:


from sklearn.metrics import confusion_matrix


# In[31]:


confusion_matrix(y_pred, y_test)


# In[ ]:




