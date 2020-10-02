#!/usr/bin/env python
# coding: utf-8

# **Gamma Log Facies Type Prediction - Sequence Labeling**
# 
# 
# *https://www.crowdanalytix.com/contests/gamma-log-facies-type-prediction*
# 

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


import pandas as pd
import numpy as np
import json
pd.options.display.max_rows = 1000
pd.options.display.max_columns = 1000
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import time

from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from sklearn.model_selection import KFold


# In[ ]:


train = pd.read_csv('/kaggle/input/gamma-log-facies/CAX_LogFacies_Train_File.csv')
print(train.shape)
train.head()


# In[ ]:


X_train = pd.pivot_table(train, values='GR', index=['well_id'], columns=['row_id'])
y_train = pd.pivot_table(train, values='label', index=['well_id'], columns=['row_id'])
X_train.shape, y_train.shape


# In[ ]:


class EarlyStopping:
    def __init__(self, patience=5, delta=0, checkpoint_path='checkpoint.pt', is_maximize=True):
        self.patience, self.delta, self.checkpoint_path = patience, delta, checkpoint_path
        self.counter, self.best_score = 0, None
        self.is_maximize = is_maximize

    def load_best_weights(self, model):
        model.load_state_dict(torch.load(self.checkpoint_path))

    def __call__(self, score, model):
        if self.best_score is None or         (score > self.best_score + self.delta if self.is_maximize else score < self.best_score - self.delta):
            torch.save(model.state_dict(), self.checkpoint_path)
            self.best_score, self.counter = score, 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    
class TabularDataset(Dataset):

    def __init__(self, df_x, df_y, is_test=False):
        self.x = df_x
        self.y = df_y
        self.n = df_x.shape[0]
        self.is_test=is_test

    def __len__(self): return self.n

    def __getitem__(self, idx): 
        if not self.is_test:
            return [self.x[idx].astype(np.float32), self.y[idx].astype(np.int64)]
        else:
            return [self.x[idx].astype(np.float32), self.y[idx]]
            

    
class Seq2SeqRnn(nn.Module):
    def __init__(self, input_size, seq_len, hidden_size, output_size, num_layers=1, bidirectional=False, dropout=.3,
            hidden_layers = [100, 200]):
        
        super().__init__()
        self.input_size = input_size
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers=num_layers
        self.bidirectional=bidirectional
        self.output_size=output_size
        
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, 
                           bidirectional=bidirectional, batch_first=True,dropout=.3)
        
         # Input Layer
        if hidden_layers and len(hidden_layers):
            first_layer  = nn.Linear(hidden_size*2 if bidirectional else hidden_size, hidden_layers[0])

            # Hidden Layers
            self.hidden_layers = nn.ModuleList(
                [first_layer]+[nn.Linear(hidden_layers[i], hidden_layers[i+1]) for i in range(len(hidden_layers) - 1)]
            )
            for layer in self.hidden_layers: nn.init.kaiming_normal_(layer.weight.data)   

            # output layers
            self.output_layer = nn.Linear(hidden_layers[-1], output_size)
            nn.init.kaiming_normal_(self.output_layer.weight.data) 
           
        else:
            self.hidden_layers = []
            self.output_layer = nn.Linear(hidden_size*2 if bidirectional else hidden_size, output_size)
            nn.init.kaiming_normal_(self.output_layer.weight.data) 
            
        self.bn_layers = nn.ModuleList([nn.BatchNorm1d(seq_len) for size in hidden_layers])
        self.activation_fn = torch.relu
            
        self.dropout = nn.Dropout(dropout)
        self.output_activation_fn = partial(torch.softmax, dim=1)
        
    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, self.seq_len, self.input_size) 
        outputs, (hidden, cell) = self.rnn(x)        

        x = self.dropout(self.activation_fn(outputs))
        for hidden_layer, bn_layer in zip(self.hidden_layers, self.bn_layers):
            x = self.activation_fn(hidden_layer(x))
#             x = bn_layer(x)
            x = self.dropout(x)
        
        x = self.output_layer(x)
#         x = self.output_activation_fn(x)
        return x


# In[ ]:


folds = KFold(n_splits=5, random_state=100, shuffle=True)
indices= [(train_index, test_index) for (train_index, test_index) in folds.split(X_train.index)]
train_index, val_index = indices[2]
len(train_index), len(val_index)


# In[ ]:


for index, (train_index, val_index ) in enumerate(indices):
    print("Fold : {}".format(index))
    
    batchsize = 16
    train_dataset = TabularDataset(df_x=X_train.iloc[train_index].values,  df_y=y_train.iloc[train_index].values)
    train_dataloader = DataLoader(train_dataset, batchsize, shuffle=True, num_workers=1)

    valid_dataset = TabularDataset(df_x=X_train.iloc[val_index].values,  df_y=y_train.iloc[val_index].values)
    valid_dataloader = DataLoader(valid_dataset, batchsize, shuffle=True, num_workers=1)

    # test_dataset = TabularDataset(df_x=X_test.values, df_y=X_test_ids.values, is_test=True)
    # test_dataloader = DataLoader(test_dataset, batchsize, shuffle=False, num_workers=1)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model=Seq2SeqRnn(input_size=1, seq_len=1100, hidden_size=256, output_size=5, num_layers=3, hidden_layers=[1024],
                     bidirectional=True).to(device)

    no_of_epochs = 100
    early_stopping = EarlyStopping(patience=10, is_maximize=True, checkpoint_path="checkpoint_{}.pt".format(index))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    schedular = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=0.001, epochs=no_of_epochs,
                                            steps_per_epoch=len(train_dataloader))
    avg_train_losses, avg_valid_losses = [], [] 


    for epoch in range(no_of_epochs):
        start_time = time.time()

        print("Epoch : {}".format(epoch))

        train_losses, valid_losses = [], []

        model.train() # prep model for training
        train_preds, train_true = torch.Tensor([]).to(device), torch.LongTensor([]).to(device)

        for x, y in train_dataloader:          
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            predictions = model(x)

            predictions_ = predictions.view(-1, predictions.shape[-1]) 
            y_ = y.view(-1)

            loss = criterion(predictions_, y_)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            schedular.step()
            # record training loss
            train_losses.append(loss.item())

            train_true = torch.cat([train_true, y_], 0)
            train_preds = torch.cat([train_preds, predictions_], 0)



        model.eval() # prep model for evaluation
        val_preds, val_true = torch.Tensor([]).to(device), torch.LongTensor([]).to(device)
        for x, y in valid_dataloader:
            x = x.to(device)
            y = y.to(device)

            predictions = model(x)
            predictions_ = predictions.view(-1, predictions.shape[-1]) 
            y_ = y.view(-1)

            loss = criterion(predictions_, y_)
            valid_losses.append(loss.item())

            val_true = torch.cat([val_true, y_], 0)
            val_preds = torch.cat([val_preds, predictions_], 0)


        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        print( "train_loss: {}, valid_loss: {}".format(train_loss, valid_loss))

        train_score = accuracy_score(train_preds.cpu().detach().numpy().argmax(1), train_true.cpu().detach().numpy())

        val_score = accuracy_score(val_preds.cpu().detach().numpy().argmax(1), val_true.cpu().detach().numpy())
        print( "train_acc: {}, valid_acc: {}".format(train_score, val_score))

        if early_stopping(val_score, model):
            print("Early Stopping...")
            print("Best Val Score: {}".format(early_stopping.best_score))
            break

        print("--- %s seconds ---" % (time.time() - start_time))
    


# In[ ]:




