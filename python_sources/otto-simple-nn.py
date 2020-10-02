#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import copy
import csv
import time
from typing import Optional, Tuple, List, Dict, Type
import lightgbm as lgb
from sklearn.metrics import log_loss
from scipy import stats

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from torch.nn.modules.module import Module
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from torch.nn.init import xavier_normal_
from torch.nn.modules.container import ModuleList
from torch.utils.data import TensorDataset, DataLoader, Dataset

torch.manual_seed(0)
np.random.seed(0)


# In[ ]:


class OttoDataset(Dataset):
    def __init__(self, data, target, idx, mode):

        self.data = data
        self.target = target
        self.idx = np.asarray(idx)
        self.mode = mode

    def __len__(self):
        return self.idx.shape[0]

    def __getitem__(self, index):
        index = self.idx[index]
        row = self.data[index]
        
        if self.mode == 'test':
            return torch.tensor(row)
        else:
            label = self.target[index]
            return torch.tensor(row), torch.tensor(label)


# In[ ]:


train = pd.read_csv('../input/otto-group-product-classification-challenge/train.csv')
test = pd.read_csv('../input/otto-group-product-classification-challenge/test.csv')
sample_submit = pd.read_csv('../input/otto-group-product-classification-challenge/sampleSubmission.csv')


# In[ ]:


train['target'] = train['target'].str.replace('Class_', '')
train['target'] = train['target'].astype(int) - 1


# In[ ]:


excluded_column = ['target', 'id']
cols = [c for c in train.columns if c not in (excluded_column + [])]

data = pd.concat([train, test]).reset_index()
for c in cols:
    data[c] = np.log(1+data[c])
data_scale = StandardScaler().fit_transform(data[cols])
data = pd.concat([data[['id', 'target']], pd.DataFrame(data_scale)], axis=1)

train = data[~data['target'].isnull()].reset_index(drop=True)
test = data[data['target'].isnull()].reset_index(drop=True)


# In[ ]:


excluded_column = ['target', 'id']
cols = [c for c in train.columns if c not in excluded_column]

test_dataset = OttoDataset(test[cols].values, np.zeros(test.shape[0]), list(range(test.shape[0])), 'test')
test_loader = torch.utils.data.DataLoader(
      dataset=test_dataset,  
      batch_size=64, 
      shuffle=False, 
      num_workers=2) 


# In[ ]:


class MLPNet (nn.Module):
    def __init__(self):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(93, 512)   
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 9)
        self.dropout1 = nn.Dropout2d(0.2)
        self.dropout2 = nn.Dropout2d(0.2)
        self.dropout3 = nn.Dropout2d(0.2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.dropout3(x)
        x = self.fc4(x)
        return x


# In[ ]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'

num_epochs = 50
 
NFOLDS = 5
RANDOM_STATE = 871972
folds = StratifiedKFold(n_splits=NFOLDS, shuffle=True, 
                        random_state=RANDOM_STATE)


# In[ ]:


y_preds = []
oof = np.zeros((len(train), 9))
for fold_n, (train_index, valid_index) in enumerate(folds.split(train, y = train['target'])):
    X_train, X_valid = train.iloc[train_index], train.iloc[valid_index]
    y_train, y_valid = X_train['target'].astype(int), X_valid['target'].astype(int)
    train_dataset = OttoDataset(X_train[cols].values, y_train.values, list(range(X_train.shape[0])), 'train')
    valid_dataset = OttoDataset(X_valid[cols].values, y_valid.values, list(range(X_valid.shape[0])), 'train')
    
    # set data loader
    train_loader = torch.utils.data.DataLoader(
          dataset=train_dataset,  
          batch_size=64, 
          shuffle=True, 
          num_workers=2) 
    valid_loader = torch.utils.data.DataLoader(
          dataset=valid_dataset,
          batch_size=64, 
          shuffle=False,
          num_workers=2)
    net = MLPNet().to(device)
    
    # optimizing
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    for epoch in range(num_epochs):
        # initialize each epoch
        train_loss, train_acc, val_loss, val_acc = 0, 0, 0, 0
        
        # ======== train_mode ======
        net.train()
        for i, (datas, labels) in enumerate(train_loader):  
            datas, labels = datas.to(device), labels.to(device)
            optimizer.zero_grad()  
            feature = net(datas.float())
            outputs = feature
            loss = criterion(outputs, labels) 
            train_loss += loss.item()
            acc = (outputs.max(1)[1] == labels).sum()
            train_acc += acc.item() 
            loss.backward()      
            optimizer.step() 
            avg_train_loss = train_loss / len(train_loader.dataset) 
            avg_train_acc = train_acc / len(train_loader.dataset) 
        # ======== valid_mode ======
        net.eval()
        with torch.no_grad():
            for datas, labels in valid_loader:        
                datas, labels = datas.to(device), labels.to(device)
                feature = net(datas.float()) 
                outputs = feature
                loss = criterion(outputs, labels) 
                val_loss += loss.item()
                acc = (outputs.max(1)[1] == labels).sum()
                val_acc += acc.item()
        avg_val_loss = val_loss / len(valid_loader.dataset)
        avg_val_acc = val_acc / len(valid_loader.dataset)
        print ('Epoch [{}/{}], Loss: {loss:.4f}, val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}' 
                       .format(epoch+1, num_epochs, i+1, loss=avg_train_loss, val_loss=avg_val_loss, val_acc=avg_val_acc))
        
    # ======== valid_mode ======
    net.eval()
    valid_predict = []
    with torch.no_grad():
        for datas, labels in valid_loader:        
            datas, labels = datas.to(device), labels.to(device)
            feature = net(datas.float()) 
            outputs = F.softmax(feature, dim=1)
            valid_predict.append(outputs)
    oof[valid_index] = torch.cat(valid_predict, dim=0)
    # ======== test ======
    test_predict = []
    with torch.no_grad():
        for datas in test_loader:        
            datas = datas.to(device)
            feature = net(datas.float())
            outputs = F.softmax(feature, dim=1)
            test_predict.append(outputs)
    y_preds.append(torch.cat(test_predict, dim=0))


# In[ ]:


column_name = ['nn_' + str(i) for i in range(9)]

pd.DataFrame(oof, columns = column_name).to_csv('oof_nn.csv', index=False)

y_pred_nn = np.zeros((len(y_preds[0]), 9))
for i in range(len(y_preds)):
    y_pred_nn += y_preds[i].cpu().numpy() / len(y_preds)
pd.DataFrame(y_pred_nn, columns = column_name).to_csv('submit_nn.csv', index=False)


# In[ ]:


submit = pd.concat([sample_submit[['id']], pd.DataFrame(y_pred_nn)], axis = 1)
submit.columns = sample_submit.columns
submit.to_csv('submit.csv', index=False)


# In[ ]:




