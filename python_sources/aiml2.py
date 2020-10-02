#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import os
print(os.listdir("../input/fruits-360_dataset/fruits-360"))


# In[7]:


import torch
import torchvision
from torchvision import transforms
from torch.utils.data import TensorDataset,DataLoader

channel_means = (0.485, 0.456, 0.406)
channel_stds = (0.229, 0.224, 0.225)
transformation = transforms.Compose([
        transforms.Resize(size=(150,150)),
        transforms.ToTensor(),
        transforms.Normalize(channel_means,channel_stds )])

batch = 64
path = "../input/fruits-360_dataset/fruits-360/"
train_dataset = torchvision.datasets.ImageFolder(path+"Training", transform=transformation)
train_loader = DataLoader(train_dataset,batch_size=batch, shuffle=True)


# In[8]:


test_val = torchvision.datasets.ImageFolder(path + "Test", transform=transformation)
test = int(len(test_val)/2)
val = int(len(test_val) - test)
val_dataset, test_dataset = torch.utils.data.random_split(test_val, [val, test])

val_loader = DataLoader(val_dataset,batch_size=batch, shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=batch, shuffle=True)

print(len(train_dataset),len(val_dataset),len(test_dataset))


# In[9]:


import torch.nn as nn
import torch.nn.functional as F
import math
class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=50, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(3),
            nn.Conv2d(in_channels=50, out_channels=64, kernel_size=4),
            nn.ReLU(),
            nn.MaxPool2d(4),
            nn.Conv2d(in_channels=64, out_channels=80, kernel_size=2),
            nn.ReLU(),
            nn.MaxPool2d(5))
        
            
        self.layer2 = nn.Sequential(
            nn.Linear(320,128), nn.ReLU(),
            nn.Linear(128,103)
        )
    def forward(self, x):
        y = self.layer2(self.layer1(x).view(x.size(0), -1))
        return y


# In[10]:


def fit(model, train_dl,val_dl, lr, epoches,tolerance):
    model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_loss = 100
    cur_tol = tolerance
    for epoche in range(epoches):
        ep_loss = 0
        for xx,yy in train_dl:
            xx,yy = xx.cuda(), yy.cuda()
            optimizer.zero_grad()
            y_pred = model(xx)
            loss = criterion(y_pred, yy)
            loss.backward()
            ep_loss+=loss.item()
            optimizer.step()
        print("Loss: {}".format(ep_loss/len(train_dl)))
        with torch.no_grad():
            val_loss=0
            for xx,yy in val_dl:
                xx,yy = xx.cuda(), yy.cuda()
                y_pred = model(xx)
                loss = criterion(y_pred, yy)
                val_loss+=loss.item()
            val_loss/=len(val_dl)
            if best_loss>= val_loss:
                best_loss = val_loss
                cur_tol = tolerance
                torch.save(model.state_dict(), "..\bestmodel.mod")
            else:
                cur_tol -= 1
            if cur_tol==0:
                model.load_state_dict(torch.load("..\bestmodel.mod"))
                break
        print("---->Val loss: {}".format(val_loss))
    print("Stop train.")
    model.cpu()


# In[11]:


net = NN()
fit(net,train_loader,val_loader,0.005,6,3)


# In[12]:


net.load_state_dict(torch.load("..\bestmodel.mod"))
y_true = []
y_pred = []
for xx,yy in test_loader:
    net.cuda()
    xx,yy = xx.cuda(), yy.cuda()
    out = net(xx).argmax(dim=1)
    y_true.extend(yy.tolist())
    y_pred.extend(out.tolist())
net.cpu()


# In[13]:


from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_true, y_pred))
print(confusion_matrix(y_true, y_pred))


# In[ ]:




