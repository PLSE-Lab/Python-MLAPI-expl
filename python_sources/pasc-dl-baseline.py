#!/usr/bin/env python
# coding: utf-8

# # **PASC DL BASELINE**
# This is the baseline model for PASC DataQuest. Due to lack of time, I have used only 1 data split for the competition out of the 40 available to achieve an accuracy around **85%-86% (dev), 88.9% (Public LB), 85.9% (Private LB)**. Using all the splits to create a CV ensemble will definitely give much better results.

# **Note about the Data used**<br>
# Processed Data splits are available at [pasc-folds](http://www.kaggle.com/rhn1903/pascfolds)<br>
# Data processing is done in a separate notebook that I won't be able to properly comment. But here's a short summary -
# * Remove rows with more than 2 outliers with IQR Rule
# * Undersample negative samples to achieve 1:1 output class distribution
# * Split dataset while maintaining 1:1 distrib[](http://)ution resulting in 8 balanced splits
# * Scale the numerical features, encode the categorical features (OHE)
# * Further use a KFold CV split to get around 40 different possible train-dev sets

# # All the imports

# In[ ]:


import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm.autonotebook import tqdm


# # Setting the hyperparameters

# In[ ]:


TRAINING_FILE = "../input/pascfolds/cleaned_8fold/train0.csv"

TRAIN_BATCH_SIZE = 128
VALID_BATCH_SIZE = 64
HIDDEN_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 100


# In[ ]:


#Standard utils from abhishek/utils
class AverageMeter:
    """
    Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    def __init__(self, patience=7, mode="max", delta=0.001):
        self.patience = patience
        self.counter = 0
        self.mode = mode
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        if self.mode == "min":
            self.val_score = np.Inf
        else:
            self.val_score = -np.Inf

    def __call__(self, epoch_score, model, model_path):

        if self.mode == "min":
            score = -1.0 * epoch_score
        else:
            score = np.copy(epoch_score)

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
            self.counter = 0

    def save_checkpoint(self, epoch_score, model, model_path):
        if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:
            print('Validation score improved ({} --> {}). Saving model!'.format(self.val_score, epoch_score))
            torch.save(model.state_dict(), model_path)
        self.val_score = epoch_score


# In[ ]:


def accuracy_score(y_pred, y_true): 
    y_pred_tag = torch.round(torch.sigmoid(y_pred))
    correct_results_sum = (y_pred_tag == y_true).sum().float()
    acc = correct_results_sum/y_true.shape[0]
    acc = torch.round(acc * 100)
    return acc


# # Creating the Model

# In[ ]:


class DoctorModel(nn.Module):
    def __init__(self, in_size, hidden_size):
        super(DoctorModel, self).__init__()
        
        self.layer_1 = nn.Linear(in_size, hidden_size) 
        self.layer_2 = nn.Linear(hidden_size, hidden_size)
        self.layer_out = nn.Linear(hidden_size, 1) 
        
        torch.nn.init.normal_(self.layer_out.weight, std=0.02)
        torch.nn.init.kaiming_normal_(self.layer_1.weight, mode='fan_in', nonlinearity='relu')
        torch.nn.init.kaiming_normal_(self.layer_2.weight, mode='fan_in', nonlinearity='relu')
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        self.batchnorm1 = nn.BatchNorm1d(hidden_size)
        self.batchnorm2 = nn.BatchNorm1d(hidden_size)
        
    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
    
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
    
        x = self.dropout(x)
        x = self.layer_out(x)
        
        return x


# # Loading the data

# In[ ]:


class DoctorDataset:
    def __init__(self, dfx):
        self.dfx = dfx
        self.X = self.dfx.iloc[:, 0:-1]
        self.Y = self.dfx.iloc[:, -1]
        
    def __len__(self):
        return len(self.dfx)
    
    def __getitem__(self, idx):
        return {
            'ids': torch.tensor(self.X.iloc[[idx]].values, dtype=torch.float).squeeze(0),
            'truth': torch.tensor(self.Y.iloc[[idx]].values, dtype=torch.float)
        }


# In[ ]:


def train_fn(data_loader, model, optimizer, criterion, device):
    model.train()
    losses = AverageMeter()
    scores = AverageMeter()
    
    tk0 = tqdm(data_loader, total=len(data_loader))
    for bi, d in enumerate(tk0):
        ids = d["ids"]
        truth = d["truth"]
        
        ids = ids.to(device, dtype=torch.float)
        truth = truth.to(device, dtype=torch.float)
        
        optimizer.zero_grad()
        pred = model(ids)
        loss = criterion(pred, truth)
        score = accuracy_score(pred, truth)
        loss.backward()
        optimizer.step()
        
        losses.update(loss.item(), ids.size(0))
        scores.update(score, ids.size(0))
        tk0.set_postfix(loss=losses.avg, score=scores.avg)


# In[ ]:


def eval_fn(data_loader, model, criterion, device):
    model.eval()
    losses = AverageMeter()
    scores = AverageMeter()
    
    with torch.no_grad():
        tk0 = tqdm(data_loader, total=len(data_loader))
        for bi, d in enumerate(tk0):
            ids = d["ids"]
            truth = d["truth"]

            ids = ids.to(device, dtype=torch.float)
            truth = truth.to(device, dtype=torch.float)

            pred = model(ids)
            loss = criterion(pred, truth)
            score = accuracy_score(pred, truth)

            losses.update(loss.item(), ids.size(0))
            scores.update(score, ids.size(0))
            tk0.set_postfix(loss=losses.avg, score=scores.avg)
        
    print(f"Accuracy: {scores.avg}")
    return scores.avg    


# # Training the model

# In[ ]:


def run(fold):
    dfx = pd.read_csv(TRAINING_FILE)
    df_train = dfx[dfx.kfold != fold].drop(["kfold"], axis=1).reset_index(drop=True)
    df_valid = dfx[dfx.kfold == fold].drop(["kfold"], axis=1).reset_index(drop=True)
    
    train_dataset = DoctorDataset(df_train)
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = TRAIN_BATCH_SIZE,
        shuffle = True,
        num_workers = 4)
    
    valid_dataset = DoctorDataset(df_valid)
    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size = VALID_BATCH_SIZE,
        shuffle = True,
        num_workers = 2)
    
    device = torch.device("cpu")
    
    model = DoctorModel(in_size=len(dfx.columns)-2, hidden_size=HIDDEN_SIZE)
    model = model.to(device)
    
    num_train_steps = int(len(df_train) / TRAIN_BATCH_SIZE * EPOCHS)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()
    criterion = criterion.to(device)
    
    es = EarlyStopping(patience=5, mode="max")
    print(f"Training is starting...")
    
    for epoch in range(EPOCHS):
        train_fn(train_data_loader, model, optimizer, criterion, device)
        score = eval_fn(valid_data_loader, model, criterion, device)
        es(score, model, model_path=f"model.bin")
        if es.early_stop:
            print("Early Stoppage")
            break


# Train

# In[ ]:


run(fold=0)


# # Evaluate
# The test data loaded here follows the same steps as described at the beginning of this notebook.

# In[ ]:


df_test = pd.read_csv("../input/pascfolds/test_all.csv")
df_test.head()


# In[ ]:


device = torch.device("cpu")

model = DoctorModel(in_size=len(df_test.columns)-1, hidden_size=HIDDEN_SIZE)
model = model.to(device)
model.load_state_dict(torch.load("model.bin", map_location=torch.device('cpu')))
model.eval()


# In[ ]:


test_dataset = DoctorDataset(df_test)
test_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size = VALID_BATCH_SIZE,
        shuffle = False,
        num_workers = 1)

final_outputs = []
with torch.no_grad():
    tk0 = tqdm(test_data_loader, total=len(test_data_loader))
    for bi, d in enumerate(tk0):
        ids = d["ids"]
        ids = ids.to(device, dtype=torch.float)

        pred = model(ids)
        pred = torch.sigmoid(pred)
  
        pred = pred.cpu().detach().numpy()
        pred = np.rint(pred).tolist()
        for p in pred:
            output = "no" if p[0] == 0.0 else "yes"
            final_outputs.append(output)


# In[ ]:


sample = pd.read_csv("../input/pasc-data-quest-20-20/sample_submission.csv")
sample.loc[:, "Y"] = final_outputs
sample.to_csv("submission.csv", index=False)


# In[ ]:


sample.head()

