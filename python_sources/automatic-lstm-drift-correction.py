#!/usr/bin/env python
# coding: utf-8

# # This is an attempt towards automatic drift correction and denoising the signals using LSTM Autoencoders.
# #####  Refered kernels:
# 1. Chris Deotte: https://www.kaggle.com/cdeotte/one-feature-model-0-930
# 1. TJ Klein: https://www.kaggle.com/friedchips/clean-removal-of-data-drift
# 1. Mobassir: https://www.kaggle.com/mobassir/understanding-ion-switching-with-modeling

# # Preprocessing
# Much of the code has been borrowed from up there, with little modifications thrown here and there :p

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score

import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

# fastai
import fastai
from fastai import *
from fastai.text import *
from fastai.callbacks import *

from tqdm.auto import tqdm
from joblib import Parallel, delayed


# In[ ]:


NNBATCHSIZE = 96
GROUP_BATCH_SIZE = 4000
SEED = 80085


# In[ ]:


def seed_all(seed_value):
    random.seed(seed_value) # Python
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = False
        
seed_all(SEED)


# In[ ]:


# plotting functions

def plot_train(train_df, label, is_train=True):
    label = f"{label} {'Training' if is_train else 'Testing'}" 
    plt.figure(figsize=(20,5)); res = 1
    plt.plot(range(0,train_df.shape[0],res),train_df.signal[0::res])
    for i in range(11): plt.plot([i*500000,i*500000],[-5,12.5],'r')
    for j in range(10): plt.text(j*500000+200000,10,str(j+1),size=20)
    plt.xlabel('Row',size=16); plt.ylabel('Signal',size=16); 
    plt.title(label + ' Data Signal - 10 batches',size=20)
    plt.show()
    
    
def plot_test(test_df, label):
    plt.figure(figsize=(20,5))
    let = ['A','B','C','D','E','F','G','H','I','J']
    r = test_df.signal.rolling(30000).mean()
    plt.plot(test_df.time.values,r)
    for i in range(21): plt.plot([500+i*10,500+i*10],[-3,6],'r:')
    for i in range(5): plt.plot([500+i*50,500+i*50],[-3,6],'r')
    for k in range(4): plt.text(525+k*50,5.5,str(k+1),size=20)
    for k in range(10): plt.text(505+k*10,4,let[k],size=16)
    plt.title(f'{label} Test Signal Rolling Mean. Has Drift wherever plot is not horizontal line',size=16)
    plt.show()


# In[ ]:


orig_df = pd.read_csv(
    '/kaggle/input/liverpool-ion-switching/train.csv', 
    dtype={'time': np.float32, 'signal': np.float32, 'open_channels':np.int32}
)

orig_test = pd.read_csv(
    '/kaggle/input/liverpool-ion-switching/test.csv', 
    dtype={'time': np.float32, 'signal': np.float32}
)

clean_df = pd.read_csv(
    '/kaggle/input/data-without-drift/train_clean.csv', 
    dtype={'time': np.float32, 'signal': np.float32, 'open_channels':np.int32}
)


# In[ ]:


# read data
def read_data(orig=False):
    sub  = pd.read_csv('/kaggle/input/liverpool-ion-switching/sample_submission.csv', dtype={'time': np.float32})
    if orig:
        train = pd.read_csv('/kaggle/input/liverpool-ion-switching/train.csv', dtype={'time': np.float32, 'signal': np.float32, 'open_channels':np.int32})
        test  = pd.read_csv('/kaggle/input/liverpool-ion-switching/test.csv', dtype={'time': np.float32, 'signal': np.float32})
    else:
        train = pd.read_csv('/kaggle/input/data-without-drift/train_clean.csv', dtype={'time': np.float32, 'signal': np.float32, 'open_channels':np.int32})
        test  = pd.read_csv('/kaggle/input/data-without-drift/test_clean.csv', dtype={'time': np.float32, 'signal': np.float32})    
    return train, test, sub

# create batches of 4000 observations
def batching(df, batch_size):
    #print(df)
    df['group'] = df.groupby(df.index//batch_size, sort=False)['signal'].agg(['ngroup']).values
    df['group'] = df['group'].astype(np.uint16)
    return df

# normalize the data (standard scaler). We can also try other scalers for a better score!
def normalize(train, test):
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    train = scaler.fit_transform(train.values.reshape(-1, 1)).flatten()
    test = scaler.transform(test.values.reshape(-1, 1)).flatten()
    
    return train, test, scaler

# get lead and lags features
def lag_with_pct_change(df, windows, prefix):
    train_cols = []
    for window in windows:    
        df[prefix + '_signal_shift_pos_' + str(window)] = df.groupby('group')['signal'].shift(window).fillna(0)
        df[prefix + '_signal_shift_neg_' + str(window)] = df.groupby('group')['signal'].shift(-1 * window).fillna(0)
        train_cols.append(prefix + '_signal_shift_pos_' + str(window))
        train_cols.append(prefix + '_signal_shift_neg_' + str(window))
    return df, train_cols

# main module to run feature engineering. Here you may want to try and add other features and check if your score imporves :).
def run_feat_engineering(df, batch_size, prefix=''):
    # create batches
    df = batching(df, batch_size=batch_size)

    # create leads and lags (1, 2, 3 making them 6 features)
    df, cols = lag_with_pct_change(df, list(range(1, 8)), prefix)
    return df, cols

def split(GROUP_BATCH_SIZE=4000, SPLITS=5, noisy=False):
    print('Reading Data Started...')
    orig_train, orig_test, sample_submission = read_data(orig=True)
    clean_train, clean_test, sample_submission = read_data(orig=False)
    
    orig_train['signal'], clean_train['signal'], scaler = normalize(orig_train['signal'], clean_train['signal'])
    print('Reading and Normalizing Data Completed')
    
    print('Feature Engineering Started...')
    orig_train, orig_train_cols = run_feat_engineering(orig_train, batch_size=GROUP_BATCH_SIZE, prefix='orig')
    clean_train = batching(clean_train, batch_size=GROUP_BATCH_SIZE)
    
    orig_train = orig_train.rename({
        'signal': 'orig_signal'
    }, axis='columns')
    clean_train = clean_train.rename({
        'signal': 'clean_signal',
    }, axis='columns')
    orig_train_cols.append('orig_signal')
    
    features = [col for col in clean_train.columns if col not in ['index', 'group', 'open_channels', 'time']]
    clean_train = clean_train[features] 
    
    joint_df = pd.concat([orig_train, clean_train], axis=1)
    print(joint_df.columns)
    print('Feature Engineering Completed...')

    X = np.array(list(joint_df.groupby('group').apply(lambda x: x[orig_train_cols].values))).astype(np.float32)
    y = np.array(list(joint_df.groupby('group').apply(lambda x: x['clean_signal'].values))).astype(np.float32)
    
    
    # preprocess the original test data too, so we can run prediction and match against clean test data.
    orig_test['signal'] = scaler.transform(orig_test['signal'].values.reshape(-1, 1)).flatten()
    orig_test, orig_test_cols = run_feat_engineering(orig_test, batch_size=GROUP_BATCH_SIZE, prefix='orig_test')
    orig_test_cols.append('signal')
    X_test = np.array(list(orig_test.groupby('group').apply(lambda x: x[orig_test_cols].values))).astype(np.float32)

    return X, np.expand_dims(y, axis=-1), X_test, clean_test, scaler


# In[ ]:


get_ipython().run_cell_magic('time', '', 'X, y, X_test, clean_test, scaler = split()')


# In[ ]:


X.shape, y.shape, X_test.shape, clean_test.shape


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED)

X_train.shape, y_train.shape


# # Dataset

# In[ ]:


from torch.utils.data import Dataset, DataLoader
class IonDataset(Dataset):
    def __init__(self, data, labels=None):
        self.data = data
        self.train_mode = labels is not None
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        
        if self.train_mode:
            labels = self.labels[idx]
            return [data.astype(np.float32), labels.astype(np.float32)]
        else:
            return data.astype(np.float32)


# # Denoising Model: Wavenet-LSTM-Autoencoder

# In[ ]:


# from https://www.kaggle.com/hanjoonchoe/wavenet-lstm-pytorch-ignite-ver        
class Wave_Block(nn.Module):
    
    def __init__(self,in_channels,out_channels,dilation_rates):
        super(Wave_Block,self).__init__()
        self.num_rates = dilation_rates
        self.convs = nn.ModuleList()
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        
        self.convs.append(nn.Conv1d(in_channels,out_channels,kernel_size=1))
        dilation_rates = [2**i for i in range(dilation_rates)]
        for dilation_rate in dilation_rates:
            self.filter_convs.append(nn.Conv1d(out_channels,out_channels,kernel_size=3,padding=dilation_rate,dilation=dilation_rate))
            self.gate_convs.append(nn.Conv1d(out_channels,out_channels,kernel_size=3,padding=dilation_rate,dilation=dilation_rate))
            self.convs.append(nn.Conv1d(out_channels,out_channels,kernel_size=1))
            
    def forward(self,x):
        x = self.convs[0](x)
        res = x
        for i in range(self.num_rates):
            x = torch.tanh(self.filter_convs[i](x)) * torch.sigmoid(self.gate_convs[i](x))
            x = self.convs[i+1](x)
            res = torch.add(res, x)
        return res
    
    
class Encoder(nn.Module):
    def __init__(self, nc, hidden_size, input_size):
        super().__init__()
        self.lstm1 = nn.GRU(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=2,
            dropout=0.2,
            batch_first=True, 
            bidirectional=True
        )
        
        self.encoder = nn.Sequential(
            Wave_Block(nc, 64, 4),
            nn.BatchNorm1d(64),
            Wave_Block(64, input_size, 1),
            nn.BatchNorm1d(input_size)
        )
        
    def forward(self, x):
        # ---- Encoder ----
        x = x.permute(0, 2, 1)
        x = self.encoder(x)
        
        # ---- Bottleneck ----
        x = x.permute(0, 2, 1)
        x, _ = self.lstm1(x)
        
        return x
    
class Model(nn.Module):
    def __init__(self, nc, hidden_size=128):
        super().__init__()
        input_size = 128

        self.encoder = Encoder(nc, hidden_size, input_size)
        
        self.decoder = nn.Sequential(
            Wave_Block(hidden_size*2, 64, 1),
            nn.BatchNorm1d(64),
            Wave_Block(64, nc, 1),
            nn.BatchNorm1d(nc)
        )
        
        self.fc = nn.Linear(nc, 1)
            
    def forward(self,x):
        x = self.encoder(x)        
        
        # ---- Decoder ----
        x = x.permute(0, 2, 1)
        x = self.decoder(x)
        
        # ---- Output ----
        x = x.permute(0, 2, 1)
        x = self.fc(x)
        
        return x


# In[ ]:


epochs = 150
folds_to_train = [0]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[ ]:


time2train = 25 * len(folds_to_train) * epochs / 3600
f"{time2train:.02f} hours"


# # Training

# In[ ]:


torch.cuda.empty_cache()

train_dataset = IonDataset(X_train, y_train)
valid_dataset = IonDataset(X_val, y_val)

nc = X_train.shape[-1]
model = Model(nc)

db = DataBunch.create(train_dataset, valid_dataset, bs=NNBATCHSIZE)
learn = Learner(db, model, loss_func=nn.MSELoss())
learn.callbacks.append(ShowGraph(learn))
learn.callbacks.append(CSVLogger(learn, filename=f"history"))
# learn.callbacks.append(SaveModelCallback(learn, name=f'model_fold_{index}'))


# In[ ]:


# learn.lr_find()
# learn.recorder.plot()
learn.fit_one_cycle(epochs, max_lr=1e-2)
learn.save('denoiser')


# # Train set denoising

# In[ ]:


def run_pred(dataset, scaler):
    pred_list = []
    pred_loader = DataLoader(IonDataset(dataset, None), NNBATCHSIZE, shuffle=False, num_workers=8, pin_memory=True)

    with torch.no_grad():
        for x in tqdm(pred_loader):
            x = x.to(device)

            predictions = model(x)
            predictions = predictions.view(-1, predictions.shape[-1])
            pred_list.extend(predictions.cpu().numpy().tolist())
            
    vals = np.array(pred_list)
    vals = scaler.inverse_transform(vals.reshape(-1, 1)).flatten()
    return vals


# In[ ]:


vals = run_pred(X, scaler)


# In[ ]:


# inverse transform to get the values back
denoised_df = orig_df.copy()
denoised_df['signal'] = vals
denoised_df.to_csv('denoised_df.csv', index=False)

from sklearn.metrics import mean_squared_error
diff = abs(clean_df['signal'] - denoised_df['signal'])
mse = mean_squared_error(clean_df['signal'], denoised_df['signal'])
print(f"Mean: {diff.mean()}, std: {diff.std()}, mse: {mse}")


# In[ ]:


plot_train(orig_df, label="Orig", is_train=True)


# In[ ]:


plot_train(clean_df, label="Cleaned", is_train=True)


# In[ ]:


plot_train(denoised_df, label="Denoised", is_train=True)


# # Test set denoising

# In[ ]:


vals = run_pred(X_test, scaler)


# In[ ]:


denoised_test_df = clean_test.copy()
denoised_test_df['signal'] = vals
denoised_test_df.to_csv('denoised_test_df.csv', index=False)


# In[ ]:


diff = abs(clean_test['signal'] - denoised_test_df['signal'])
mse = mean_squared_error(clean_test['signal'], denoised_test_df['signal'])
print(f"Mean: {diff.mean()}, std: {diff.std()}, mse: {mse}")


# In[ ]:


plot_test(orig_test, label="Orig")


# In[ ]:


plot_test(clean_test, label="Clean")


# In[ ]:


plot_test(denoised_test_df, label="Denoised")

