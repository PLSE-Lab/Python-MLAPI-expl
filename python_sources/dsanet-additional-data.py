#!/usr/bin/env python
# coding: utf-8

# ## DSANet approach
# 
# Solution based on these implementations [1](https://github.com/bighuang624/DSANet) & [2](https://www.kaggle.com/kirichenko17roman/old-dsanet-approach) of [Dual Self-Attention Network for Multivariate Time Series Forecasting](https://dl.acm.org/doi/10.1145/3357384.3358132)
# 
# Additional information about the contries is added before the dense layer.
# 
# NN architecture:
# 
# ![](https://raw.githubusercontent.com/bighuang624/DSANet/master/docs/DSANet-model-structure.png)

# Define week number:

# In[ ]:


week = 4


# Load libraries:

# In[ ]:


import numpy as np 
import pandas as pd
import datetime
import os
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 500)


# In[ ]:


train = pd.read_csv(f"/kaggle/input/covid19-global-forecasting-week-{week}/train.csv")
train["geo"] = np.where(train["Province_State"].isna(), train["Country_Region"], train["Country_Region"] + "_" + train["Province_State"])

train['Date'] = pd.to_datetime(train['Date'])
train = train.loc[train['Date'] > '2020-02-20', :]
train_last_date = train.Date.unique()[-1]
print(f"Dataset has training data untill : {str(train_last_date)[:10]}")
print(f"Training dates: {len(train.Date.unique())}")


# In[ ]:


train.info()


# In[ ]:


train.describe(include='all')


# In[ ]:


additional = pd.read_csv(f"../input/covid19-country-data-wk3-release/Data Join - RELEASE.csv")
additional.info()


# In[ ]:


additional["TRUE POPULATION"] = additional["TRUE POPULATION"].str.strip().str.replace(',','').astype(int)
additional['pct_in_largest_city'] = additional['pct_in_largest_city'].str.replace('%','').astype(float)
additional[' TFR '] = additional[' TFR '].replace('N.A.',np.nan).astype(float)
additional['Personality_uai'] = pd.to_numeric(additional['Personality_uai'], errors='coerce')
additional['Personality_pdi'] = pd.to_numeric(additional['Personality_pdi'], errors='coerce')
additional['Personality_idv'] = pd.to_numeric(additional['Personality_idv'], errors='coerce')
additional['Personality_ltowvs'] = pd.to_numeric(additional['Personality_ltowvs'], errors='coerce')
additional['personality_agreeableness'] = pd.to_numeric(additional['personality_agreeableness'], errors='coerce')
additional['AIR_AVG'] = pd.to_numeric(additional['AIR_AVG'], errors='coerce')
additional[' Avg_age '] = pd.to_numeric(additional[' Avg_age '], errors='coerce')
additional['Personality_mas'] = pd.to_numeric(additional['Personality_mas'], errors='coerce')


# In[ ]:


additional.info()


# In[ ]:



additional["geo"] = np.where(additional["Province_State"].isna(), additional["Country_Region"], additional["Country_Region"] + "_" + additional["Province_State"])
additional.describe(include ='all')


# In[ ]:


additional.describe()


# In[ ]:


additional.drop(['Province_State','Country_Region'],inplace=True,axis=1)


# In[ ]:


additional.info()


# In[ ]:


train_add = train.merge(additional,on='geo',how='left')


# In[ ]:


train_add.describe(include='all')


# In[ ]:


from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from functools import partial
from sklearn.feature_selection import mutual_info_regression, SelectKBest
from sklearn.ensemble import RandomForestRegressor
## Class for feature Selection
class selectFeaturesTransformer(BaseEstimator, TransformerMixin):
    """Custom scaling transformer"""
    def __init__(self, k=10,method='RF',discreteCol=[], order=[],scores=[]):
        """ 
        initialize transformer
        Inputs : 
            k -- number of features to keep
            method -- method to use, either 'Mutual Information or RF
            discreteCol -- if Mutual Information is used, specify indexes of discrete columns
        """
        self.k = k
        self.method = method
        self.order = order
        self.discreteCol = discreteCol
        self.scores = scores
        
        
        

    def fit(self, X_train,y_train):
        """
        Fit the transformer on data
        Input :
            X_train -- features array
            Y_train -- labels array
        Output :
            fitted transformer
        """
        if self.method == "Mutual Information" :
            
            if len(y_train.shape)>1 and y_train.shape[1]>1 :
                scores = np.zeros(X_train.shape[1])
                for i in y_train.columns :
                    discrete_mutual_info_regression = partial(mutual_info_regression,discrete_features=self.discreteCol)
                    featS = SelectKBest(k=self.k, score_func=discrete_mutual_info_regression).fit(X_train,y_train[i] )
                    scores += featS.scores_
                    print("Top 10 selected by Mutual information for ",i)
                    print(list(X_train.columns[np.flip(featS.scores_.argsort())]))
                self.order = np.flip(scores.argsort())
                self.scores = np.flip(np.sort(scores))
            else :
                discrete_mutual_info_regression = partial(mutual_info_regression)
                featS = SelectKBest(k=self.k, score_func=discrete_mutual_info_regression).fit(X_train,y_train )
                self.order = np.flip(featS.scores_.argsort())
                self.scores = np.flip(np.sort(featS.scores_))
            #self.selectedColumns = [columns_eng[i]  for i in self.order[:self.k]]
            #return X_train[:,order_mi[:self.k]]
        
        elif self.method == 'RF' :
            if len(y_train.shape)>1 and y_train.shape[1]>1 :
                scores = np.zeros(X_train.shape[1])
                for i in y_train.columns :
                    rfModel = RandomForestRegressor(n_estimators=500,random_state =0).fit(X_train.values, y_train[i].values)
                    scores = scores + rfModel.feature_importances_
                    print("Top 10 selected by Random Forest for ",i)
                    print(list(X_train.columns[np.flip(rfModel.feature_importances_.argsort())]))
                self.order = np.flip(scores.argsort())
                self.scores = np.flip(np.sort(scores))
            else :        
                rfModel = RandomForestRegressor(random_state =0).fit(X_train, y_train)
                order = np.flip(rfModel.feature_importances_.argsort())
                self.order = np.flip(rfModel.feature_importances_.argsort())
                self.scores = np.flip(np.sort(rfModel.feature_importances_))
            #self.selectedColumns = [columns_eng[i]  for i in order_rf[:self.k]]
            #return X_train[:,order_[:self.k]]
        return self
            
                
        
    def transform(self, X_train):
        """
        apply fitted transformer to select features
        Input :
            X_train -- features array
        Output :
            array containing only selected features
        """
        return X_train[self.order]


# In[ ]:


add_cols = [c for c in train_add.columns if c not in train.columns]


# In[ ]:


#discreteCol = [i for i in range(len(X_train.columns)) if X_train_norm.columns[i] in ['year','month','day','dayYear']]
X_train = train_add[add_cols]
X_train.replace(np.nan,X_train.mean(),inplace=True)
Fs = selectFeaturesTransformer(method="Mutual Information", discreteCol=[],k=len(X_train.columns),order=[])
Fs.fit(X_train,train_add['ConfirmedCases'])
print("Top 10 selected by Mutual information for ConfirmedCases ")
print(list(X_train.columns[Fs.order]))


# In[ ]:


print(list(Fs.scores))


# In[ ]:


Fs_rf = selectFeaturesTransformer(method="RF", discreteCol=[],k=X_train.shape[1],order=[])
Fs_rf.fit(X_train.drop(['geo'],axis=1),train_add['ConfirmedCases'])
print("Top 10 selected by Random Forest for ConfirmedCases ")
print(list(X_train.drop(['geo'],axis=1).columns[Fs_rf.order]))


# In[ ]:


print(list(Fs_rf.scores))


# In[ ]:


Fs_rf = selectFeaturesTransformer(method="RF", discreteCol=[],k=X_train.shape[1],order=[])
Fs_rf.fit(X_train,train_add['Fatalities'])
print("Top 10 selected by Random Forest for ConfirmedCases ")
print(list(X_train.columns[Fs_rf.order]))


# In[ ]:


print(list(Fs_rf.scores))


# In[ ]:


import seaborn as sns
plt.figure(figsize=(30,30))
corr = X_train.corr()
fig, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corr, vmax=.8,annot=True, square=True)


# In[ ]:


to_use = ['GDP_region', 'max_high_rises', 'pct_in_largest_city', 'TRUE POPULATION', 'latitude', 
          'personality_perform', 'Personality_idv', 'Personality_ltowvs', 'AIR_AVG', 
          'Personality_mas', 'Personality_pdi', 'continent_Life_expectancy', 'personality_agreeableness', 
          'Personality_uai', 'continent_happiness', 'continent_corruption', 'continent_generosity',
           'longitude', 'murder', 'humidity', ' TFR ',
           'temperature', 'Personality_assertive']


# In[ ]:


test = pd.read_csv(f"/kaggle/input/covid19-global-forecasting-week-{week}/test.csv")
test['Date'] = pd.to_datetime(test['Date'])
test_first_date = test['Date'].values[0]
test_last_date = test['Date'].values[-1]
print(f'Test period from {str(test_first_date)[:10]} to {str(test_last_date)[:10]}')


# In[ ]:


period = (np.array(test_last_date, dtype='datetime64[D]').astype(np.int64) - np.array(train_last_date, dtype='datetime64[D]').astype(np.int64))


# In[ ]:


print(f"Prediction days: {(np.array(test_last_date, dtype='datetime64[D]').astype(np.int64) - np.array(train_last_date, dtype='datetime64[D]').astype(np.int64))+1}")
print(f"Public set: {(np.array(train_last_date, dtype='datetime64[D]').astype(np.int64) - np.array(test_first_date, dtype='datetime64[D]').astype(np.int64))+1}")
print(f"Full prediction set: {(np.array(test_last_date, dtype='datetime64[D]').astype(np.int64) - np.array(test_first_date, dtype='datetime64[D]').astype(np.int64))+1}")


# Data window for forecast:

# In[ ]:


win = 20


# In[ ]:


base_1 = train.pivot(index='Date', columns="geo", values='ConfirmedCases').iloc[-win,:].values
base_2 = train.pivot(index='Date', columns="geo", values='Fatalities').iloc[-win,:].values


# In[ ]:


train.pivot(index='geo', columns="Date", values=['ConfirmedCases']).values


# In[ ]:


train


# I use only new cases and new death's dynamics to make the prediction.

# In[ ]:


train['ConfirmedCases'] = train['ConfirmedCases'] - train.groupby('geo')['ConfirmedCases'].shift(periods=1)
train['Fatalities'] = train['Fatalities'] - train.groupby('geo')['Fatalities'].shift(periods=1)

train = train.groupby('geo').tail(train.groupby('geo').size().values[0]-1)

train['ConfirmedCases'] = np.where(train['ConfirmedCases'] < 0, 0.0, train['ConfirmedCases'])
train['Fatalities'] = np.where(train['Fatalities'] < 0, 0.0, train['Fatalities'])
train


# Select only March and April:

# In[ ]:


train_add['geo']


# In[ ]:


add_d


# In[ ]:


X_train['geo'] = train_add['geo']


# In[ ]:


X_train.describe(include='all')


# In[ ]:



X = X_train.set_index('geo')
X.loc[~X.index.duplicated(keep='first')]
add_d = X.loc[~X.index.duplicated(keep='first')][to_use]


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
add_d = scaler.fit_transform(add_d)


# In[ ]:


train_cases = train.pivot(index='Date', columns="geo", values='ConfirmedCases').iloc[:-3,:].values
valid_cases = train.pivot(index='Date', columns="geo", values='ConfirmedCases').iloc[-(win+3):,:].values

train_fatal = train.pivot(index='Date', columns="geo", values='Fatalities').iloc[:-3,:].values
valid_fatal = train.pivot(index='Date', columns="geo", values='Fatalities').iloc[-(win+3):,:].values


# In[ ]:


train_cases.shape


# In[ ]:


_ = plt.plot(train_cases)


# In[ ]:


_ = plt.plot(valid_cases)


# In[ ]:


_ = plt.plot(train_fatal)


# In[ ]:


_ = plt.plot(valid_fatal)


# ## Model

# We need to install pytorch lightning library:

# In[ ]:


get_ipython().run_cell_magic('bash', '', '\npip install pytorch_lightning')


# In[ ]:


import torch.nn as nn
import torch
import torch.nn.functional as F
import pytorch_lightning as ptl

from torch import optim
from torch.utils.data import DataLoader
from collections import OrderedDict


# Loss function:

# In[ ]:


def rmsle(predict, target): 
    return torch.sqrt(((torch.log(predict + 1) - torch.log(target + 1))**2).mean())


# Data loader:

# In[ ]:


torch.from_numpy(np.array([1,2,3])).type(torch.FloatTensor)


# In[ ]:


class MTSFDataset(torch.utils.data.Dataset):

    def __init__(self, window, horizon, set_type, tra, validation,add_tr,add_val):
        
        assert type(set_type) == type('str')
        
        self.window = window
        self.horizon = horizon
        self.tra = tra
        self.validation = validation
        self.set_type = set_type
        
        if set_type == 'train':
            rawdata = tra
            self.add = torch.from_numpy(add_tr).type(torch.FloatTensor)
        elif set_type == 'validation':
            rawdata = validation
            self.add = torch.from_numpy(add_val).type(torch.FloatTensor)

        self.len, self.var_num = rawdata.shape
        self.sample_num = max(self.len - self.window - self.horizon + 1, 0)
        self.samples, self.labels = self.__getsamples(rawdata)

    def __getsamples(self, data):
        X = torch.zeros((self.sample_num, self.window, self.var_num))
        Y = torch.zeros((self.sample_num, 1, self.var_num))
       # print('################',self.sample_num,self.var_num,'#####################')
        for i in range(self.sample_num):
            start = i
            end = i + self.window
            X[i, :, :] = torch.from_numpy(data[start:end, :])
            Y[i, :, :] = torch.from_numpy(data[end+self.horizon-1, :])
        
        return (X, Y)

    def __len__(self):
        return self.sample_num

    def __getitem__(self, idx):
        sample = [self.samples[idx, :, :], self.labels[idx, :, :],self.add]
        #print("######## Sample #####", sample[0].shape,sample[1].shape,sample[2].shape)

        return sample


# In[ ]:


class ScaledDotProductAttention(nn.Module):

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn

class MultiHeadAttention(nn.Module):

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)


    def forward(self, q, k, v):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)

        output, attn = self.attention(q, k, v)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn

class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1)
        self.w_2 = nn.Conv1d(d_hid, d_in, 1)
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output

class EncoderLayer(nn.Module):

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input)

        enc_output = self.pos_ffn(enc_output)

        return enc_output, enc_slf_attn


class DecoderLayer(nn.Module):

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, dec_input, enc_output, non_pad_mask=None, slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_output, dec_slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input, mask=slf_attn_mask)

        dec_output, dec_enc_attn = self.enc_attn(
            dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)

        dec_output = self.pos_ffn(dec_output)

        return dec_output, dec_slf_attn, dec_enc_attn


class Single_Global_SelfAttn_Module(nn.Module):

    def __init__(
            self,
            window, n_multiv, n_kernels, w_kernel,
            d_k, d_v, d_model, d_inner,
            n_layers, n_head, drop_prob=0.1):
        '''
        Args:
        window (int): the length of the input window size
        n_multiv (int): num of univariate time series
        n_kernels (int): the num of channels
        w_kernel (int): the default is 1
        d_k (int): d_model / n_head
        d_v (int): d_model / n_head
        d_model (int): outputs of dimension
        d_inner (int): the inner-layer dimension of Position-wise Feed-Forward Networks
        n_layers (int): num of layers in Encoder
        n_head (int): num of Multi-head
        drop_prob (float): the probability of dropout
        '''

        super(Single_Global_SelfAttn_Module, self).__init__()

        self.window = window
        self.w_kernel = w_kernel
        self.n_multiv = n_multiv
        self.d_model = d_model
        self.drop_prob = drop_prob
        self.conv2 = nn.Conv2d(1, n_kernels, (window, w_kernel))
        self.in_linear = nn.Linear(n_kernels, d_model)
        self.out_linear = nn.Linear(d_model, n_kernels)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=drop_prob)
            for _ in range(n_layers)])

    def forward(self, x, return_attns=False):

        x = x.view(-1, self.w_kernel, self.window, self.n_multiv)
        x2 = F.relu(self.conv2(x))
        x2 = nn.Dropout(p=self.drop_prob)(x2)
        x = torch.squeeze(x2, 2)
        x = torch.transpose(x, 1, 2)
        src_seq = self.in_linear(x)

        enc_slf_attn_list = []

        enc_output = src_seq

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attn_list
        enc_output = self.out_linear(enc_output)
        return enc_output,


class Single_Local_SelfAttn_Module(nn.Module):

    def __init__(
            self,
            window, local, n_multiv, n_kernels, w_kernel,
            d_k, d_v, d_model, d_inner,
            n_layers, n_head, drop_prob=0.1):
        '''
        Args:
        window (int): the length of the input window size
        n_multiv (int): num of univariate time series
        n_kernels (int): the num of channels
        w_kernel (int): the default is 1
        d_k (int): d_model / n_head
        d_v (int): d_model / n_head
        d_model (int): outputs of dimension
        d_inner (int): the inner-layer dimension of Position-wise Feed-Forward Networks
        n_layers (int): num of layers in Encoder
        n_head (int): num of Multi-head
        drop_prob (float): the probability of dropout
        '''

        super(Single_Local_SelfAttn_Module, self).__init__()

        self.window = window
        self.w_kernel = w_kernel
        self.n_multiv = n_multiv
        self.d_model = d_model
        self.drop_prob = drop_prob
        self.conv1 = nn.Conv2d(1, n_kernels, (local, w_kernel))
        self.pooling1 = nn.AdaptiveMaxPool2d((1, n_multiv))
        self.in_linear = nn.Linear(n_kernels, d_model)
        self.out_linear = nn.Linear(d_model, n_kernels)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=drop_prob)
            for _ in range(n_layers)])

    def forward(self, x, return_attns=False):

        x = x.view(-1, self.w_kernel, self.window, self.n_multiv)
        x1 = F.relu(self.conv1(x))
        x1 = self.pooling1(x1)
        x1 = nn.Dropout(p=self.drop_prob)(x1)
        x = torch.squeeze(x1, 2)
        x = torch.transpose(x, 1, 2)
        src_seq = self.in_linear(x)

        enc_slf_attn_list = []

        enc_output = src_seq

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attn_list
        enc_output = self.out_linear(enc_output)
        return enc_output,

class AR(nn.Module):

    def __init__(self, window):
        super(AR, self).__init__()
        self.linear = nn.Linear(window, 1)

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        x = self.linear(x)
        x = torch.transpose(x, 1, 2)
        return x

class DSANet(ptl.LightningModule):

    def __init__(self, tra, validation, add_t, add_v, n_multiv, batch_size=16, window=64, local=3, n_kernels=32, 
                 drop_prob=0.1, criterion='rmsle_loss', learning_rate=0.005, horizon=14):
        
        super(DSANet, self).__init__()

        self.batch_size = batch_size

        self.window = window
        self.local = local
        self.n_multiv = n_multiv
        self.n_kernels = n_kernels
        self.w_kernel = 1
        self.n_add = add_t.shape[1]

        self.d_model = 512
        self.d_inner = 2048
        self.n_layers = 6
        self.n_head = 8
        self.d_k = 64
        self.d_v = 64
        self.drop_prob = drop_prob

        self.criterion = criterion
        self.learning_rate = learning_rate
        self.horizon = horizon
        self.tra = tra
        self.validation = validation
        self.add_t = add_t
        self.add_v = add_v
        
        self.losses_v = []
        self.losses_t = []

        self.__build_model()

    def __build_model(self):

        self.sgsf = Single_Global_SelfAttn_Module(
            window=self.window, n_multiv=self.n_multiv, n_kernels=self.n_kernels,
            w_kernel=self.w_kernel, d_k=self.d_k, d_v=self.d_v, d_model=self.d_model,
            d_inner=self.d_inner, n_layers=self.n_layers, n_head=self.n_head, drop_prob=self.drop_prob)

        self.slsf = Single_Local_SelfAttn_Module(
            window=self.window, local=self.local, n_multiv=self.n_multiv, n_kernels=self.n_kernels,
            w_kernel=self.w_kernel, d_k=self.d_k, d_v=self.d_v, d_model=self.d_model,
            d_inner=self.d_inner, n_layers=self.n_layers, n_head=self.n_head, drop_prob=self.drop_prob)

        self.ar = AR(window=self.window)
        self.W_output1 = nn.Linear(2 * self.n_kernels+self.n_add, 2 * self.n_kernels+self.n_add)
        self.W_output2 = nn.Linear(2 * self.n_kernels+self.n_add, 1)
        self.dropout = nn.Dropout(p=self.drop_prob)
        self.active_func = nn.Tanh()

    def forward(self, x, add):
 
        sgsf_output, *_ = self.sgsf(x)
        slsf_output, *_ = self.slsf(x)
        sf_output = torch.cat((sgsf_output, slsf_output, add), 2)
        #print('#######',sf_output.shape)
        
        #print('#######',sf_output.shape)
        sf_output = self.dropout(sf_output)
        sf_output = self.W_output1(sf_output)
        #print('#######',sf_output.shape)
        sf_output = self.W_output2(sf_output)
        #print('#######',sf_output.shape)

        sf_output = torch.transpose(sf_output, 1, 2)

        ar_output = self.ar(x)

        output = sf_output + ar_output
        output[output < 0] = 0.0

        return output

    def loss(self, labels, predictions):
        if self.criterion == 'l1_loss':
            loss = F.l1_loss(predictions, labels)
        elif self.criterion == 'mse_loss':
            loss = F.mse_loss(predictions, labels)
        elif self.criterion == 'rmsle_loss':
            loss = rmsle(predictions, labels)
        return loss

    def training_step(self, data_batch, batch_i):

        x, y, add = data_batch

        y_hat = self.forward(x, add)

        loss_val = self.loss(y, y_hat)

        if self.trainer.use_dp:
            loss_val = loss_val.unsqueeze(0)

        output = OrderedDict({
            'loss': loss_val
        })
        self.losses_t.append(torch.mean(loss_val))
        return output

    def validation_step(self, data_batch, batch_i):

        x, y, add = data_batch

        y_hat = self.forward(x, add)

        loss_val = self.loss(y, y_hat)

        if self.trainer.use_dp:
            loss_val = loss_val.unsqueeze(0)

        output = OrderedDict({
            'val_loss': loss_val,
            'y': y,
            'y_hat': y_hat,
        })
        self.losses_v.append(torch.mean(loss_val))
        return output

    def validation_epoch_end(self, outputs):

        loss_sum = 0
        for x in outputs:
            loss_sum += x['val_loss'].item()
        val_loss_mean = loss_sum / len(outputs)

        y = torch.cat(([x['y'] for x in outputs]), 0)
        y_hat = torch.cat(([x['y_hat'] for x in outputs]), 0)

        num_var = y.size(-1)
        y = y.view(-1, num_var)
        y_hat = y_hat.view(-1, num_var)
        sample_num = y.size(0)

        y_diff = y_hat - y
        y_mean = torch.mean(y)
        y_translation = y - y_mean

        val_rrse = torch.sqrt(torch.sum(torch.pow(y_diff, 2))) / torch.sqrt(torch.sum(torch.pow(y_translation, 2)))

        y_m = torch.mean(y, 0, True)
        y_hat_m = torch.mean(y_hat, 0, True)
        y_d = y - y_m
        y_hat_d = y_hat - y_hat_m
        corr_top = torch.sum(y_d * y_hat_d, 0)
        corr_bottom = torch.sqrt((torch.sum(torch.pow(y_d, 2), 0) * torch.sum(torch.pow(y_hat_d, 2), 0)))
        corr_inter = corr_top / corr_bottom
        val_corr = (1. / num_var) * torch.sum(corr_inter)

        val_mae = (1. / (sample_num * num_var)) * torch.sum(torch.abs(y_diff))

        tqdm_dic = {
            'val_loss': val_loss_mean,
            'RRSE': val_rrse.item(),
            'CORR': val_corr.item(),
            'MAE': val_mae.item()
        }
        return tqdm_dic

    def configure_optimizers(self):

        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler] 

    def __dataloader(self, train):

        set_type = train
        dataset = MTSFDataset(window=self.window, horizon=self.horizon,
                              set_type=set_type, 
                              tra=self.tra, validation=self.validation,add_tr=self.add_t, add_val=self.add_v)

        train_sampler = None
        batch_size = self.batch_size

        should_shuffle = train_sampler is None
        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=should_shuffle,
            sampler=train_sampler,
            num_workers=4
        )

        return loader

    @ptl.data_loader
    def train_dataloader(self):
        return self.__dataloader(train='train')

    @ptl.data_loader
    def val_dataloader(self):
        return self.__dataloader(train='validation')


# ## New confirmed cases prediction

# In[ ]:


add_d[:,0:1].shape


# In[ ]:


#rom p.loggers import TensorBoardLogger
#logger = ptl.loggers.TensorBoardLogger("tb_logs", name="model_bis")
model_cases = DSANet(train_cases, valid_cases,add_d,add_d, train_cases.shape[1], window=win, learning_rate=0.005, horizon=1, drop_prob=0.2)

trainer = ptl.Trainer(val_check_interval=1, max_steps=200,gpus=1)#,logger=logger) 
trainer.fit(model_cases) 


# In[ ]:


plt.plot(model_cases.losses_t,label = 'T')
plt.plot(model_cases.losses_v,label='V')
plt.legend()


# In[ ]:


plt.plot(model_cases.losses_t,label = 'T')
plt.plot(model_cases.losses_v,label='V')
plt.legend()


# In[ ]:


torch.mean(model_cases.losses_v)


# In[ ]:





# In[ ]:


from glob import glob

sd = torch.load(glob("/kaggle/working/lightning_logs/version_8/checkpoints/*.ckpt")[0])
model_cases.load_state_dict(sd['state_dict'])


# In[ ]:


add_cuda.shape


# In[ ]:


input = train.pivot(index='Date', columns="geo", values='ConfirmedCases').iloc[-win:,:].values

for i in range(period):
    ins = torch.tensor(input[-win:, :]).cuda()
    add_cuda = torch.tensor([add_d]).cuda()
    pred = model_cases(ins.unsqueeze(dim=0).float(),add_cuda.float())
    
    input = np.concatenate([input, np.array(pred.detach().cpu().numpy(), dtype=np.int).reshape(1, train_cases.shape[1])], axis=0)


# In[ ]:


train_cases.shape


# ## New fatal cases prediction

# In[ ]:


model_fatal = DSANet(train_fatal, valid_fatal, add_d,add_d, train_fatal.shape[1], window=win, learning_rate=0.0005, horizon=1, drop_prob=0.2)

trainer = ptl.Trainer(val_check_interval=1, max_steps=200,gpus=1) #
trainer.fit(model_fatal) 


# In[ ]:


sd = torch.load(glob("/kaggle/working/lightning_logs/version_11/checkpoints/*.ckpt")[0])
model_fatal.load_state_dict(sd['state_dict'])


# In[ ]:


input2 = train.pivot(index='Date', columns="geo", values='Fatalities').iloc[-win:,:].values

for i in range(period):
    
    ins = torch.tensor(input2[-win:, :]).cuda()
    add_cuda = torch.tensor([add_d]).cuda()
    pred = model_fatal(ins.unsqueeze(dim=0).float(),add_cuda.float())
    
    input2 = np.concatenate([input2, np.array(pred.detach().cpu().numpy(), dtype=np.int).reshape(1, train_fatal.shape[1])], axis=0)


# ## Forecast preparation

# In[ ]:


pred_size = (np.array(test_last_date, dtype='datetime64[D]').astype(np.int64) - np.array(test_first_date, dtype='datetime64[D]').astype(np.int64))+1


# In[ ]:


pd.DataFrame(np.array(input.cumsum(0) + base_1, dtype=np.int)[-pred_size:,:], columns=train.pivot(index='Date', columns="geo", values='ConfirmedCases').columns).loc[:, ['US_New York', 'Ukraine', 'Italy', 'Spain']]


# In[ ]:


pd.DataFrame(np.array(input2.cumsum(0) + base_2, dtype=np.int)[-pred_size:,:], columns=train.pivot(index='Date', columns="geo", values='ConfirmedCases').columns).loc[:, ['US_New York', 'Ukraine', 'Italy', 'Spain']]


# Convert predicted new cases to total cases:

# In[ ]:


input = input.cumsum(0) + base_1
input2 = input2.cumsum(0) + base_2


# In[ ]:


import datetime 

def prov(i):
    try:
        return i.split("_")[1]
    except:
        return None

res = pd.DataFrame(input2[-pred_size:,:], columns=train.pivot(index='Date', columns="geo", values='Fatalities').columns).unstack().reset_index(name='Fatalities')     .merge(
    pd.DataFrame(input[-pred_size:,:], columns=train.pivot(index='Date', columns="geo", values='ConfirmedCases').columns).unstack().reset_index(name='ConfirmedCases'),
          how='left', on=['geo', 'level_1']
)

res['Date'] = [test.Date[0] + datetime.timedelta(days=i) for i in res['level_1']]
res['Province_State'] = [prov(i) for i in res['geo']]
res['Country_Region'] = [i.split("_")[0] for i in res['geo']]

res


# Adjust results and create submission:

# In[ ]:


sub = pd.read_csv(f"/kaggle/input/covid19-global-forecasting-week-{week}/submission.csv")

sub = test.merge(res, how='left', on=['Date', 'Province_State', 'Country_Region']).loc[:, ["ForecastId", "ConfirmedCases", "Fatalities"]]
sub['Fatalities'] = np.where(sub['Fatalities']*7 > sub["ConfirmedCases"], sub["ConfirmedCases"] / 7, sub['Fatalities']) 
sub["ConfirmedCases"] = np.where(((sub['Fatalities'] / sub["ConfirmedCases"]) < 0.0005) & (sub["ConfirmedCases"] > 1000), sub['Fatalities']*2000, sub["ConfirmedCases"])

sub['Fatalities'] = np.array(sub['Fatalities'], dtype=np.int)
sub["ConfirmedCases"] = np.array(sub["ConfirmedCases"], dtype=np.int)

sub


# Write prediction:

# In[ ]:


sub.to_csv("submission.csv", index=False)


# In[ ]:




