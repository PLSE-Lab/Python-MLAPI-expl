#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install ipdb')
get_ipython().system('pip install pytorch_lightning')


# In[ ]:



import numpy as np
import pandas as pd 
import holidays
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
import xgboost as xgb
import holidays
import shap
import datetime as dt
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import plotly.offline as py
from IPython.display import FileLink
import seaborn as sns
pd.plotting.register_matplotlib_converters()


py.init_notebook_mode()
FileLink('__notebook_source__.ipynb')


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


data = pd.read_csv('/kaggle/input/chicago-divvy-bicycle-sharing-data/data.csv')
data['starttime'] = pd.to_datetime(data['starttime'])
data['stoptime'] = pd.to_datetime(data['stoptime'])
data = data.sort_values('starttime').set_index('starttime')
data['events'] = data.events.fillna('unknown')


# ## Initial Analysis and Visualizations
# 
# Starting with a basic graph of daily trip counts and the daily high temperature we see that the two seem fairly highly correlated.

# In[ ]:


daily_df = data.resample('1d').agg({'tripduration':'count','temperature':'max'})
fig, ax = plt.subplots(figsize=(8,6))
daily_df['tripduration'].plot(ax=ax, label='Daily Trips')
ax2 = ax.twinx()
daily_df['temperature'].plot(ax=ax2, color='orange', label='Daily High Temperature')
ax.legend(bbox_to_anchor=(1.1,1))
ax2.legend(bbox_to_anchor=(1.1,0.95))
ax.set_ylabel(r"Total Trips")
ax2.set_ylabel(r"Daily High ($^\circ$F)")
ax.set_title('Daily Trips vs Daily High')


# To start to get a sense of the correlation between the two, we look at the rolling 30 day correlation between total daily trips and the daily high temperature. We see that the correlation seems highest (around 0.8) in the spring and fall each year and dips during the summer months.

# In[ ]:


daily_corr = daily_df[['tripduration','temperature']].rolling(30).corr().iloc[0::2,-1].reset_index().set_index('starttime')
fig, ax = plt.subplots(figsize=(8,6))
daily_df['tripduration'].rolling(30).mean().plot(ax=ax, label='Daily Trips 30d MA')
ax2 = ax.twinx()
daily_corr['temperature'].plot(ax=ax2, color='orange', label='Trips-Temperature 30d Correlation')
ax.legend(bbox_to_anchor=(1.1,1))
ax2.legend(bbox_to_anchor=(1.1,0.95))
ax.set_ylabel(r"Total Trips")
ax2.set_ylabel(r"Rolling 30d Correlation")
ax.set_title('Trips vs Temperature Correlation Rolling')


# Now we look whether it is important if we look at the total trips completed or the total trip length. The heatmap below shows the correlation by hour of day and day of week between total trips and the total trip length. Generally they are very highly correlated so we will stick to total trips completed for the rest of the analysis.

# In[ ]:


daily_df_trip_count = data.resample('1h').agg({'tripduration':'count'})
daily_df_trip_count = daily_df_trip_count.rename(columns={'tripduration':'trip_count'})
daily_df_trip_sum = data.resample('1h').agg({'tripduration':'sum'})
daily_df_trip_sum = daily_df_trip_sum.rename(columns={'tripduration':'trip_sum'})
daily_df_trip_sum = daily_df_trip_sum.merge(daily_df_trip_count, left_index=True, right_index=True,how='inner')
daily_df_trip_sum['Week Day'] = daily_df_trip_sum.index.weekday_name
daily_df_trip_sum['Hour'] = daily_df_trip_sum.index.hour

trip_length_count_corr = daily_df_trip_sum.groupby(['Hour','Week Day'])['trip_sum','trip_count'].corr().reset_index()
cats = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
trip_length_count_corr['Week Day'] = pd.Categorical(trip_length_count_corr['Week Day'], categories=cats, ordered=True)

fig, ax = plt.subplots(figsize=(8,6))
hmap = sns.heatmap(pd.pivot_table(trip_length_count_corr[trip_length_count_corr['trip_sum']!=1],
                                  index='Week Day',columns='Hour', values='trip_sum'),
                   cmap='viridis', cbar_kws={'label': 'Correlation'})
ax.set_title('Correlation between Trips Length and Trip Count by Hour and Day of Week')


# No we look at a heatmap of the correlation between Temperature and Total Trips by hour of day and day of week. We see that the correlation is lowest during the commuting times (especially the morning commute) and lowest on evenings and weekends, which intuitively make sense.

# In[ ]:


data['Week Day'] = data.index.weekday_name
hourly_correlation = data.groupby(['hour','Week Day'])['tripduration','temperature'].corr().reset_index()
cats = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
hourly_correlation['Week Day'] = pd.Categorical(hourly_correlation['Week Day'], categories=cats, ordered=True)

fig, ax = plt.subplots(figsize=(8,6))
hmap = sns.heatmap(pd.pivot_table(hourly_correlation[hourly_correlation['tripduration']!=1],
                                  index='Week Day',columns='hour', values='tripduration'),
                   cmap='viridis', cbar_kws={'label': 'Correlation'})
ax.set_title('Correlation between Trips and Temperature by Hour and Day of Week')


# ## Modelling
# 
# No we will examine a few different models to see how well we can estimate hourly bike trips for each gender (I split on gender to create more training data) and how important temperature is as a variable. 
# 
# Below I start by creating various features as well as taking the log transform of the hourly trips and then standardizing. 
# 
# Finally I split the data into three partitions:
# * Train (2015-02-01 : 2017-07-14)
# * Validation (2017-07-15 : 2017-08-31) 
# * Test (2017-09-01: 2017-12-31) 

# In[ ]:


hourly = data.groupby('gender').resample('1h').agg({'tripduration':'count','temperature':'max','events':'first'})
hourly['temperature'] = hourly['temperature'].interpolate() 
hourly['temperature'] = (hourly['temperature'] - hourly['temperature'].mean())/hourly['temperature'].std()

hourly['tripduration'] = np.log(hourly['tripduration']+1)
hourly['tripduration'] = (hourly['tripduration'] - hourly['tripduration'].mean())/hourly['tripduration'].std()

hourly['lag_7'] = hourly.tripduration.shift(7*24)
hourly['lag_1'] = hourly.tripduration.shift(24)
hourly['yoy'] = hourly.tripduration.rolling(24*30).mean().pct_change(365*24).rolling(3*24).mean().shift(1)

hourly['lag_14'] = hourly.tripduration.shift(14*24)
hourly['lag_28'] = hourly.tripduration.shift(28*24)
hourly['ma_28'] = hourly.tripduration.rolling(24*28).mean().shift(1)
hourly['ma_7'] = hourly.tripduration.rolling(7*24).mean().shift(1)
hourly['ma_1'] = hourly.tripduration.rolling(24).mean().shift(1)
hourly['events'] = hourly['events'].fillna(method='ffill') 

hourly=hourly.reset_index()
hourly['date'] = pd.to_datetime(hourly.starttime.dt.date)

daily_temp = data.resample('1d').agg({'temperature':'max'}).reset_index()
daily_temp['date'] = pd.to_datetime(daily_temp.starttime.dt.date)
daily_temp = daily_temp.rename(columns={'temperature':'daily_max_temp'})


daily_temp['daily_max_temp'] = (daily_temp['daily_max_temp'] - daily_temp['daily_max_temp'].mean())/daily_temp['daily_max_temp'].std()

holiday_df = pd.DataFrame(holidays.US(years = [2014,2015,2016,2017]).items(), columns =['date','holiday'] )
holiday_df['date'] = pd.to_datetime(holiday_df['date'])

hourly = hourly.merge(holiday_df, on='date', how='left')
hourly = hourly.merge(daily_temp[['date','daily_max_temp']], on='date', how='left')

hourly['holiday_indicator'] = hourly['holiday'].notnull().astype(int)

hourly['day'] = hourly.starttime.dt.day-1
hourly['hour'] = hourly.starttime.dt.hour

hourly['wday'] = hourly.starttime.dt.weekday
hourly['month'] = hourly.starttime.dt.month-1


hourly['holiday_indicator_lag_minus_2'] = hourly.holiday_indicator.shift(-2)
hourly['holiday_indicator_lag_minus_1'] = hourly.holiday_indicator.shift(-1)
hourly['holiday_indicator_lag_1'] = hourly.holiday_indicator.shift(1)
hourly['holiday_indicator_lag_2'] = hourly.holiday_indicator.shift(2)


hourly['year'] = hourly.starttime.dt.year - 2014
hourly['gender_enc'] = LabelEncoder().fit_transform(hourly['gender'])
hourly['holiday_enc'] = LabelEncoder().fit_transform(hourly['holiday'].fillna('None'))
hourly['events_enc'] = LabelEncoder().fit_transform(hourly['events'])

train_data = hourly[(hourly['date']<pd.to_datetime('2017-07-15')) & (hourly['date']>pd.to_datetime('2015-02-01'))]
val_data = hourly[(hourly['date']>=pd.to_datetime('2017-07-15'))&(hourly['date']<pd.to_datetime('2017-09-01'))]
test_data = hourly[(hourly['date']>pd.to_datetime('2017-09-01'))]


# ### XGBoost
# 
# Xgboost is usually a solid baseline and we can use the SHAP package to get nice visualizations of various features effects on model outputs. 

# In[ ]:


cols = ['temperature','wday','holiday_enc','events_enc','year','lag_7', 'lag_14',
       'ma_28', 'ma_7','hour','holiday_indicator_lag_minus_2',
       'holiday_indicator_lag_minus_1', 'holiday_indicator_lag_1',
       'holiday_indicator_lag_2','gender_enc', 'daily_max_temp','yoy']

params = {"learning_rate": 0.1,
          'objective': 'reg:linear',
          'max_depth':4,
          'eval_metric':'rmse','verbose':0}

dtrain = xgb.DMatrix(train_data[cols].values, label=train_data['tripduration'])
dval = xgb.DMatrix(val_data[cols].values, label=val_data['tripduration'])

watchlist = [(dval, 'eval')]
num_round = 500

model = xgb.train(params,dtrain, num_round , watchlist, early_stopping_rounds=20,verbose_eval=50)
test_data['y_pred'] = model.predict(xgb.DMatrix(test_data[cols].values))


# We look at the predictions on the test data to evaluate how effective the model is. With a RMSE of 0.2 we would expect it to be pretty good.

# In[ ]:


test_data[test_data.gender=='Male'].set_index('starttime')[['y_pred','tripduration']].loc[pd.to_datetime('2017-09-01'):pd.to_datetime('2017-09-07')].plot()


# In[ ]:


test_data[test_data.gender=='Female'].set_index('starttime')[['y_pred','tripduration']].loc[pd.to_datetime('2017-09-01'):pd.to_datetime('2017-09-07')].plot()


# In the following two plot we resample daily to see how well the model works over the entire testing horizon - pretty well. 

# In[ ]:


test_data[test_data.gender=='Female'].set_index('starttime')[['y_pred','tripduration']].loc[pd.to_datetime('2017-09-01'):pd.to_datetime('2017-12-31')].resample('1d').sum().plot()


# In[ ]:


test_data[test_data.gender=='Male'].set_index('starttime')[['y_pred','tripduration']].loc[pd.to_datetime('2017-09-01'):pd.to_datetime('2017-12-31')].resample('1d').sum().plot()


# #### Now we can use the SHAP Package to examine the feature effects on the model output. 
# 
# The SHAP methodology is based on 'Shapely Values' from Game Theory. We take take the average model output and the estimate the additive contributions from each feature to arrive at the final model output for each record. 

# In[ ]:


shap.initjs()
explainer = shap.TreeExplainer(model, feature_perturbation='interventional' )
samples = train_data[cols].sample(2000)
shap_values = explainer.shap_values(samples.values)


# Overall the 7 day and 14 day lag are the most important, but both temperature and daily high temperature are fairly important as well. 

# In[ ]:


shap.summary_plot(shap_values, samples)


# When we look at the effect of temperature on model output we see a fairly linear relationship. 

# In[ ]:


shap.dependence_plot("temperature", shap_values, samples)


# For daily high temperature we see the contributions are more significantly effected for colder temperatures versus warmer temperatures. This confirms what we saw with rolling 30 day correlation between trips and temperature being lower in the summer.  

# In[ ]:


shap.dependence_plot("daily_max_temp", shap_values, samples)


# Final we see what we would expect for the hourly effects, peaks at rush hour and lowest in the early morning hours. 

# In[ ]:


shap.dependence_plot("hour", shap_values, samples)


# ## Temporal Fusion Transformer Model

# The TFT model was presented in the paper: *Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting* available here: https://arxiv.org/abs/1912.09363. The architecture is shown below. I ported it (not exactly but closely) from TensorFlow to PyTorch and thought it would be interesting to see how it works on this bike share data. While this model is definitely overkill for the problem here, I thought it would be interesting to compare and see how 'interpretable' it is using the weights that come out of the 'Variable Selection Network' modules of the model. I use the Pytorch Lightning framework for the model training.
# 
# I start by creating the same train, validation and test sets, but this time since the model takes in whole sequences I create a sequence based on 21 days of hourly data, of which 14 days will be used for the encoder and the task it to predict the next 7 days of hourly data. To do so we using a sliding window approach to create slices of 21 days of data over the training and validation datasets. 

# In[ ]:


from IPython.display import Image 
Image("/kaggle/input/tftarch/tft_arch.png")


# In[ ]:


get_ipython().system('mkdir training_data')
get_ipython().system('mkdir validation_data')
get_ipython().system('mkdir test_data')


# In[ ]:


date_ranges = train_data.date.unique()

x_cols = ['tripduration', 'temperature', 'daily_max_temp','ma_7', 'lag_7','yoy']

identifier_cols = ['gender_enc']

time_columns =['year','day', 'events_enc',
       'holiday_enc','wday','hour']

all_cols = x_cols + time_columns
seq_len = 21

batch = 0

count = 1
for i in range(len(date_ranges)-1, 21, -1):
    
    
    d_cols = date_ranges[i-seq_len: i]
    if (i == (len(date_ranges)-1)):
        x_vals = np.nan_to_num(np.array(train_data[train_data.date.isin(d_cols)].groupby('gender')[x_cols].apply(lambda x : x.values.tolist()).to_list()))
        identifiers = np.array(train_data[train_data.date == d_cols[0]].groupby('gender')[identifier_cols].apply(lambda x : x.values.tolist()).to_list())
        time_covariates = np.array(train_data[train_data.date.isin(d_cols)].groupby('gender')[time_columns].apply(lambda x : x.values.tolist()).to_list())
    
    else: 
        x_vals = np.append(x_vals, np.nan_to_num(np.array(train_data[train_data.date.isin(d_cols)].groupby('gender')[x_cols].apply(lambda x : x.values.tolist()).to_list())), axis=0)
        identifiers = np.append(identifiers, np.array(train_data[train_data.date == d_cols[0]].groupby('gender')[identifier_cols].apply(lambda x : x.values.tolist()).to_list()), axis=0)
        time_covariates = np.append(time_covariates, np.array(train_data[train_data.date.isin(d_cols)].groupby('gender')[time_columns].apply(lambda x : x.values.tolist()).to_list()), axis=0)

path = 'training_data/'

np.save('{}time_covariates.npy'.format(path),time_covariates)
np.save('{}x_vals.npy'.format(path),x_vals)
np.save('{}identifiers.npy'.format(path),identifiers)


# In[ ]:


date_ranges = val_data.date.unique()
x_cols = ['tripduration', 'temperature','daily_max_temp','ma_7', 'lag_7','yoy']

identifier_cols = ['gender_enc']

time_columns =['year','day', 'events_enc',
       'holiday_enc','wday','hour']


batch = 0

count = 1
for i in range(len(date_ranges)-1, seq_len, -1):
    
    
    d_cols = date_ranges[i-seq_len: i]
    if (i == (len(date_ranges)-1)):
        x_vals = np.nan_to_num(np.array(val_data[val_data.date.isin(d_cols)].groupby('gender')[x_cols].apply(lambda x : x.values.tolist()).to_list()))
        identifiers = np.array(val_data[val_data.date == d_cols[0]].groupby('gender')[identifier_cols].apply(lambda x : x.values.tolist()).to_list())
        time_covariates = np.array(val_data[val_data.date.isin(d_cols)].groupby('gender')[time_columns].apply(lambda x : x.values.tolist()).to_list())
    
    else: 
        x_vals = np.append(x_vals, np.nan_to_num(np.array(val_data[val_data.date.isin(d_cols)].groupby('gender')[x_cols].apply(lambda x : x.values.tolist()).to_list())), axis=0)
        identifiers = np.append(identifiers, np.array(val_data[val_data.date == d_cols[0]].groupby('gender')[identifier_cols].apply(lambda x : x.values.tolist()).to_list()), axis=0)
        time_covariates = np.append(time_covariates, np.array(val_data[val_data.date.isin(d_cols)].groupby('gender')[time_columns].apply(lambda x : x.values.tolist()).to_list()), axis=0)

path = 'validation_data/'

np.save('{}time_covariates.npy'.format(path),time_covariates)
np.save('{}x_vals.npy'.format(path),x_vals)
np.save('{}identifiers.npy'.format(path),identifiers)


# In[ ]:


date_ranges = test_data.date.unique()
x_cols = ['tripduration', 'temperature','daily_max_temp','ma_7', 'lag_7','yoy']

identifier_cols = ['gender_enc']

time_columns =['year','day', 'events_enc',
       'holiday_enc','wday','hour']


batch = 0

count = 1
for i in range(len(date_ranges)-1, seq_len, -1):
    
    
    d_cols = date_ranges[i-seq_len: i]
    if (i == (len(date_ranges)-1)):
        x_vals = np.nan_to_num(np.array(test_data[test_data.date.isin(d_cols)].groupby('gender')[x_cols].apply(lambda x : x.values.tolist()).to_list()))
        identifiers = np.array(test_data[test_data.date == d_cols[0]].groupby('gender')[identifier_cols].apply(lambda x : x.values.tolist()).to_list())
        time_covariates = np.array(test_data[test_data.date.isin(d_cols)].groupby('gender')[time_columns].apply(lambda x : x.values.tolist()).to_list())
    
    else: 
        x_vals = np.append(x_vals, np.nan_to_num(np.array(test_data[test_data.date.isin(d_cols)].groupby('gender')[x_cols].apply(lambda x : x.values.tolist()).to_list())), axis=0)
        identifiers = np.append(identifiers, np.array(test_data[test_data.date == d_cols[0]].groupby('gender')[identifier_cols].apply(lambda x : x.values.tolist()).to_list()), axis=0)
        time_covariates = np.append(time_covariates, np.array(test_data[test_data.date.isin(d_cols)].groupby('gender')[time_columns].apply(lambda x : x.values.tolist()).to_list()), axis=0)

path = 'test_data/'

np.save('{}time_covariates.npy'.format(path),time_covariates)
np.save('{}x_vals.npy'.format(path),x_vals)
np.save('{}identifiers.npy'.format(path),identifiers)


# ### Model code and DataLoader (for the training/validation samples we just created)

# In[ ]:


class BikeShareDataset(Dataset):

    def __init__(self, path='training_data/'):
        
    
            self.x = np.load('{}x_vals.npy'.format(path))
            self.linspace = np.expand_dims(np.concatenate([np.linspace(-1,0, 24*14), np.linspace(0.05,1, 24*7)], axis=0), -1)
            self.tv = np.load('{}time_covariates.npy'.format(path))
            self.ids = np.load('{}identifiers.npy'.format(path))
          
    def __getitem__(self, index):
        
        return {'x': np.concatenate([self.x[index],self.linspace], axis=1) , 'dates':self.tv[index], 'id':self.ids[index].squeeze()}
    
    def __len__(self):
        
        return self.x.shape[0]


# In[ ]:


from torch import nn
import math
import torch


class TimeDistributed(nn.Module):
    ## Takes any module and stacks the time dimension with the batch dimenison of inputs before apply the module
    ## From: https://discuss.pytorch.org/t/any-pytorch-function-can-work-as-keras-timedistributed/1346/4
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y

class GLU(nn.Module):
    #Gated Linear Unit
    def __init__(self, input_size):
        super(GLU, self).__init__()
        
        self.fc1 = nn.Linear(input_size,input_size)
        self.fc2 = nn.Linear(input_size, input_size)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        
        sig = self.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return torch.mul(sig, x)


class GatedResidualNetwork(nn.Module):
    def __init__(self, input_size, hidden_state_size, output_size, dropout, hidden_context_size=None, batch_first=False):
        super(GatedResidualNetwork, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_context_size = hidden_context_size
        self.hidden_state_size=hidden_state_size
        self.dropout = dropout
        
        if self.input_size!=self.output_size:
            self.skip_layer = TimeDistributed(nn.Linear(self.input_size, self.output_size))

        self.fc1 = TimeDistributed(nn.Linear(self.input_size, self.hidden_state_size), batch_first=batch_first)
        self.elu1 = nn.ELU()
        
        if self.hidden_context_size is not None:
            self.context = TimeDistributed(nn.Linear(self.hidden_context_size, self.hidden_state_size),batch_first=batch_first)
            
        self.fc2 = TimeDistributed(nn.Linear(self.hidden_state_size,  self.output_size), batch_first=batch_first)
        self.elu2 = nn.ELU()
        
        self.dropout = nn.Dropout(self.dropout)
        self.bn = TimeDistributed(nn.BatchNorm1d(self.output_size),batch_first=batch_first)
        self.gate = TimeDistributed(GLU(self.output_size), batch_first=batch_first)

    def forward(self, x, context=None):

        if self.input_size!=self.output_size:
            residual = self.skip_layer(x)
        else:
            residual = x
        
        x = self.fc1(x)
        if context is not None:
            context = self.context(context)
            x = x+context
        x = self.elu1(x)
        
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.gate(x)
        x = x+residual
        x = self.bn(x)
        
        return x


class VariableSelectionNetwork(nn.Module):
    def __init__(self, input_size, num_inputs, hidden_size, dropout, context=None):
        super(VariableSelectionNetwork, self).__init__()

        self.hidden_size = hidden_size
        self.input_size =input_size
        self.num_inputs = num_inputs
        self.dropout = dropout
        self.context=context

        if self.context is not None:
            self.flattened_grn = GatedResidualNetwork(self.num_inputs*self.input_size, self.hidden_size, self.num_inputs, self.dropout, self.context)
        else:
            self.flattened_grn = GatedResidualNetwork(self.num_inputs*self.input_size, self.hidden_size, self.num_inputs, self.dropout)


        self.single_variable_grns = nn.ModuleList()
        for i in range(self.num_inputs):
            self.single_variable_grns.append(GatedResidualNetwork(self.input_size, self.hidden_size, self.hidden_size, self.dropout))

        self.softmax = nn.Softmax(dim=2)

    def forward(self, embedding, context=None):

        if context is not None:
            sparse_weights = self.flattened_grn(embedding, context)
        else:
            sparse_weights = self.flattened_grn(embedding)

        sparse_weights = self.softmax(sparse_weights).unsqueeze(2)

        var_outputs = []
        for i in range(self.num_inputs):
            ##select slice of embedding belonging to a single input
            var_outputs.append(self.single_variable_grns[i](embedding[:,:, (i*self.input_size) : (i+1)*self.input_size]))

        var_outputs = torch.stack(var_outputs, axis=-1)

        outputs = sparse_weights*var_outputs
        
        outputs = outputs.sum(axis=-1)

        return outputs, sparse_weights


# In[ ]:


"""
Implementation of Temporal Fusion Transformers: https://arxiv.org/abs/1912.09363
"""
from torch import nn
import math
import torch
import ipdb

from torch.utils.data import DataLoader,Dataset
import pytorch_lightning as pl


class TFT(pl.LightningModule):

    def __init__(self, hparams):
        super(TFT, self).__init__()

        self.device = hparams['device']
        self.batch_size = hparams['batch_size']
        self.static_variables = hparams['static_variables']
        self.encode_length = hparams['encode_length']
        self.time_varying_categoical_variables =  hparams['time_varying_categoical_variables']
        self.time_varying_real_variables_encoder =  hparams['time_varying_real_variables_encoder']
        self.time_varying_real_variables_decoder =  hparams['time_varying_real_variables_decoder']
        self.static_embedding_vocab_sizes =  hparams['static_embedding_vocab_sizes']
        self.time_varying_embedding_vocab_sizes =  hparams['time_varying_embedding_vocab_sizes']
        self.num_input_series_to_mask = hparams['num_masked_series']
        self.hidden_size = hparams['lstm_hidden_dimension']
        self.lstm_layers = hparams['lstm_layers']
        self.dropout = hparams['dropout']
        self.embedding_dim = hparams['embedding_dim']
        self.attn_heads = hparams['attn_heads']
        self.seq_length = hparams['seq_length']
        self.learning_rate =hparams['learning_rate']
        self.train_dl = hparams['train_dl']
        self.val_dl = hparams['val_dl']
        self.mse_loss = nn.MSELoss()
        
        self.static_embedding_layers = nn.ModuleList()
        for i in range(self.static_variables):
            emb = nn.Embedding(self.static_embedding_vocab_sizes[i], self.embedding_dim).to(self.device)
            self.static_embedding_layers.append(emb)
        
        
        self.time_varying_embedding_layers = nn.ModuleList()
        for i in range(self.time_varying_categoical_variables):
            emb = TimeDistributed(nn.Embedding(self.time_varying_embedding_vocab_sizes[i], self.embedding_dim), batch_first=True).to(self.device)
            self.time_varying_embedding_layers.append(emb)

            
        self.time_varying_linear_layers = nn.ModuleList()
        for i in range(self.time_varying_real_variables_encoder):
            emb = TimeDistributed(nn.Linear(1, self.embedding_dim), batch_first=True).to(self.device)
            self.time_varying_linear_layers.append(emb)

        self.encoder_variable_selection = VariableSelectionNetwork(self.embedding_dim,
                                (self.time_varying_real_variables_encoder +  self.time_varying_categoical_variables),
                                self.hidden_size,
                                self.dropout,
                                self.embedding_dim*self.static_variables)

        self.decoder_variable_selection = VariableSelectionNetwork(self.embedding_dim,
                                (self.time_varying_real_variables_decoder +  self.time_varying_categoical_variables),
                                self.hidden_size,
                                self.dropout,
                                self.embedding_dim*self.static_variables)

        
        self.lstm_encoder_input_size = self.embedding_dim*(self.time_varying_real_variables_encoder +  
                                                        self.time_varying_categoical_variables +
                                                        self.static_variables)
        
        self.lstm_decoder_input_size = self.embedding_dim*(self.time_varying_real_variables_decoder +  
                                                        self.time_varying_categoical_variables +
                                                        self.static_variables)
                                      

        self.lstm_encoder = nn.LSTM(input_size=self.hidden_size, 
                            hidden_size=self.hidden_size,
                           num_layers=self.lstm_layers,
                           dropout=self.dropout)
        
        self.lstm_decoder = nn.LSTM(input_size=self.hidden_size,
                                   hidden_size=self.hidden_size,
                                   num_layers=self.lstm_layers,
                                   dropout=self.dropout)

        self.post_lstm_gate = TimeDistributed(GLU(self.hidden_size))
        self.post_lstm_norm = TimeDistributed(nn.BatchNorm1d(self.hidden_size))

        self.static_enrichment = GatedResidualNetwork(self.hidden_size,self.hidden_size, self.hidden_size, self.dropout, self.embedding_dim*self.static_variables)
        
        self.multihead_attn = nn.MultiheadAttention(self.hidden_size, self.attn_heads)
        self.post_attn_gate = TimeDistributed(GLU(self.hidden_size))

        self.post_attn_norm = TimeDistributed(nn.BatchNorm1d(self.hidden_size, self.hidden_size))
        self.pos_wise_ff = GatedResidualNetwork(self.hidden_size, self.hidden_size, self.hidden_size, self.dropout)

        self.pre_output_norm = TimeDistributed(nn.BatchNorm1d(self.hidden_size, self.hidden_size))
        self.pre_output_gate = TimeDistributed(GLU(self.hidden_size))

        self.single_output_layer = TimeDistributed(nn.Linear(self.hidden_size, 1), batch_first=True)

        
    def init_hidden(self, batch_size):
        return torch.zeros(self.lstm_layers, batch_size, self.hidden_size, device=self.device)
        
    def apply_embedding(self, x, static_embedding, dates, apply_masking):
        ###x should have dimensions (batch_size, timesteps, input_size)
        ## Apply masking is used to mask variables that should not be accessed after the encoding steps

        #Time-varying real valued embeddings 
        if apply_masking:
            time_varying_real_vectors = []
            for i in range(self.time_varying_real_variables_decoder):
                emb = self.time_varying_linear_layers[i+self.num_input_series_to_mask](x[:,:,i+self.num_input_series_to_mask].view(x.size(0), -1, 1))
                time_varying_real_vectors.append(emb)
            time_varying_real_embedding = torch.cat(time_varying_real_vectors, dim=2)

        else: 
            time_varying_real_vectors = []
            for i in range(self.time_varying_real_variables_encoder):
                emb = self.time_varying_linear_layers[i](x[:,:,i].view(x.size(0), -1, 1))
                time_varying_real_vectors.append(emb)
            time_varying_real_embedding = torch.cat(time_varying_real_vectors, dim=2)
        
        
        ##Time-varying categorical embeddings (ie hour)
        time_varying_categoical_vectors = []
        for i in range(self.time_varying_categoical_variables):
            emb = self.time_varying_embedding_layers[i](dates[:, :,i].view(x.size(0), -1, 1).long())
            time_varying_categoical_vectors.append(emb)
        
        time_varying_categoical_embedding = torch.cat(time_varying_categoical_vectors, dim=2)  

        ##repeat static_embedding for all timesteps
        static_embedding = static_embedding.unsqueeze(1)
        static_embedding = static_embedding.repeat(1, time_varying_categoical_embedding.size(1), 1 )
        
        ##concatenate all embeddings
        embeddings = torch.cat([static_embedding,time_varying_categoical_embedding,time_varying_real_embedding], dim=2)
        
        ##emddings are returned in (time_steps, batch_size, num_variables)
        return embeddings.view(-1, embeddings.size(0), embeddings.size(2))
    
    def encode(self, x, hidden=None):
    
        if hidden is None:
            hidden = self.init_hidden(x.size(1))
            
        output, (hidden, cell) = self.lstm_encoder(x, (hidden, hidden))
        
        return output, hidden
        
    def decode(self, x, hidden=None):
        
        if hidden is None:
            hidden = self.init_hidden(x.size(1))
            
        output, (hidden, cell) = self.lstm_decoder(x, (hidden,hidden))
        
        return output, hidden


    def forward(self, x, identifiers, dates, verbose=False):

        ##x inputs should be in this order
            # target (will be masked in the decoder)
            # additional variables not known in the future (ie. temperature - will also be masked in the decoder)
            # variable known in past and future

        embedding_vectors = []
        for i in range(self.static_variables):
            emb = self.static_embedding_layers[i](identifiers[:,i].long())
            embedding_vectors.append(emb)



        ##Embedding and variable selection
        static_embedding = torch.cat(embedding_vectors, dim=1)
        embeddings_encoder = self.apply_embedding(x[:,:self.encode_length,:].float().to(self.device), 
                                                  static_embedding,
                                                  dates[:,:self.encode_length,:].float(),
                                                  apply_masking=False)
        
        embeddings_decoder = self.apply_embedding(x[:,self.encode_length:,:].float().to(self.device), 
                                                  static_embedding,
                                                  dates[:,self.encode_length:,:].float(),
                                                  apply_masking=True)

        ##static embedding (end of the embeddings_encoder and embeddings_decoder tensors) are passed as the context vectors
        embeddings_encoder, encoder_sparse_weights = self.encoder_variable_selection(embeddings_encoder[:,:,:-(self.embedding_dim*self.static_variables)],
                                                                                     embeddings_encoder[:,:,-(self.embedding_dim*self.static_variables):])
        embeddings_decoder, decoder_sparse_weights = self.decoder_variable_selection(embeddings_decoder[:,:,:-(self.embedding_dim*self.static_variables)],
                                                                                     embeddings_decoder[:,:,-(self.embedding_dim*self.static_variables):])

        
        ##LSTM
        lstm_input = torch.cat([embeddings_encoder,embeddings_decoder], dim=0)
        encoder_output, hidden = self.encode(embeddings_encoder)
        decoder_output, _ = self.decode(embeddings_decoder, hidden)
        lstm_output = torch.cat([encoder_output, decoder_output], dim=0)

        ##skip connection over lstm
        lstm_output = self.post_lstm_gate(lstm_output+lstm_input)

        ##static enrichment
        static_embedding = static_embedding.unsqueeze(0)
        static_embedding = static_embedding.repeat(lstm_output.size(0),1, 1)
        attn_input = self.static_enrichment(lstm_output, static_embedding)

        ##skip connection over lstm
        attn_input = self.post_lstm_norm(lstm_output)

        ##Decoder Attention
        attn_output, attn_output_weights = self.multihead_attn(attn_input[self.encode_length:,:,:], attn_input[:self.encode_length,:,:], attn_input[:self.encode_length,:,:])

        ##skip connection over attention
        attn_output = self.post_attn_gate(attn_output) + attn_input[self.encode_length:,:,:]
        attn_output = self.post_attn_norm(attn_output)

        output = self.pos_wise_ff(attn_output)

        ##skip connection over Decoder
        output = self.pre_output_gate(output) + lstm_output[self.encode_length:,:,:]

        #Final output layers
        output = self.pre_output_norm(output)
        single_output = self.single_output_layer(output.view(x.size(0), -1, self.hidden_size))

        if verbose:
            return single_output,encoder_output, decoder_output, attn_output, attn_output_weights, encoder_sparse_weights, decoder_sparse_weights
        else:
            return single_output

    def training_step(self, batch, batch_idx):
        x, y = batch['x'].float(), batch['x'][:,self.encode_length:,0].float()
        
        categorical_vals = batch['id'].squeeze(1).long()
        time_covs = batch['dates'].long()

        y_hat = self.forward(x, categorical_vals, time_covs)
        
        rmseloss = torch.sqrt(self.mse_loss(y_hat.squeeze(), y.flatten(1)))

        tensorboard_logs = {'train_loss': rmseloss}

        return {'loss': rmseloss, 'log': tensorboard_logs}
    
    def validation_step(self, batch, batch_idx):

        x, y = batch['x'].float(), batch['x'][:,self.encode_length:,0].float()
        
        categorical_vals = batch['id'].squeeze(1).long()
        time_covs = batch['dates'].long()

        y_hat = self.forward(x, categorical_vals, time_covs)
        
        rmseloss = torch.sqrt(self.mse_loss(y_hat.squeeze(), y.flatten(1)))
        return {'val_loss': rmseloss} 
    
    def validation_end(self, outputs):

        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs, 'progress_bar': {'val_loss': avg_loss}}

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, patience=2, verbose=True, min_lr=1e-7)  # note early stopping has patient 3
        return [optim],[scheduler]

    def train_dataloader(self):
        return DataLoader(self.train_dl, batch_size=self.batch_size, drop_last=False, shuffle=True, num_workers=4)

    def val_dataloader(self):      
        return DataLoader(self.val_dl, batch_size=self.batch_size, drop_last=False, shuffle=True, num_workers=4)

        


# In[ ]:


training_data = BikeShareDataset(path = 'training_data/')
validation_data = BikeShareDataset(path = 'validation_data/')
static_embedding_vocab_sizes = []
for i in range(training_data.ids.shape[-1]):
    static_embedding_vocab_sizes.append(training_data.ids[:,:,i].max()+1)
print('Static Embedding Sizes: ' + str(static_embedding_vocab_sizes))

time_varying_embedding_vocab_sizes = []
for i in range(training_data.tv.shape[-1]):
    time_varying_embedding_vocab_sizes.append(training_data.tv[:,:,i].max()+1)
print('Time Varying Embedding Sizes: ' + str(time_varying_embedding_vocab_sizes))


# In[ ]:


config = {}
config['static_variables'] = 1
config['time_varying_categoical_variables'] = 6
config['time_varying_real_variables_encoder'] = 7
config['time_varying_real_variables_decoder'] = 6
config['num_masked_series'] = 1
config['static_embedding_vocab_sizes'] = static_embedding_vocab_sizes
config['time_varying_embedding_vocab_sizes'] = time_varying_embedding_vocab_sizes
config['embedding_dim'] = 8
config['lstm_hidden_dimension'] = 64
config['lstm_layers'] = 1
config['dropout'] = 0.0
config['device'] = 'cuda'
config['batch_size'] = 32
config['encode_length'] = 24*14
config['seq_length'] = 24*21
config['learning_rate'] = 0.005
config['attn_heads'] = 2
config['train_dl'] = training_data
config['val_dl'] = validation_data


# In[ ]:


model = TFT(config)


# In[ ]:


from pytorch_lightning import Trainer

trainer = Trainer(max_epochs=20, gpus=1, progress_bar_refresh_rate=4)    
trainer.fit(model) 


# In[ ]:


testing_data = BikeShareDataset(path = 'test_data/')

test_DL = DataLoader(testing_data, batch_size=64, drop_last=False, shuffle=True, num_workers=1)

for batch in test_DL:
    break


# In[ ]:


x, y = batch['x'].float().cuda(), batch['x'][:,model.encode_length:,0].float()
        
categorical_vals = batch['id'].squeeze(1).long().cuda()
time_covs = batch['dates'].long().cuda()

output,encoder_output, decoder_output,     attn_output, attn_output_weights,     encoder_sparse_weights, decoder_sparse_weights = model.forward(x, categorical_vals, time_covs, verbose=True)


# In[ ]:


index_select = 24
pd_test_data = test_data[[x.all() for x in  (test_data[['year','day', 'events_enc',
       'holiday_enc','wday','hour']].values == time_covs[index_select][0].cpu().numpy())]]

pd_test_data  = pd_test_data[pd_test_data.gender_enc == categorical_vals[index_select][0].cpu().item()]
date_range = pd.date_range(start=pd_test_data.starttime.values[0], periods=504, freq='h')

test_data_slice = test_data[(test_data.starttime.isin(date_range)) & (test_data.gender_enc == categorical_vals[index_select][0].cpu().item())].reset_index()
test_data_slice['y_pred_tft'] =  pd.Series(np.concatenate([batch['x'][0,:model.encode_length,0].cpu().numpy(), output[index_select, :,0].detach().cpu().numpy()]))


# ### TFT Output
# Below we compare two sets of predections - one from above (xgboost) and the ones produced by the TFT model and include the RMSE on the slice shown below. The XGBoost model is slightly more accurate.  

# In[ ]:


test_data_slice['starttime'] = pd.to_datetime(test_data_slice['starttime'])
test_data_slice=  test_data_slice.set_index('starttime')
test_data_slice[['tripduration','y_pred','y_pred_tft']].plot(figsize=(8,6))


rmse_xgb = ((test_data_slice.iloc[24*14:]['tripduration'] - test_data_slice.iloc[24*14:]['y_pred'])** 2).mean() **.5
rmse_tft = ((test_data_slice.iloc[24*14:]['tripduration'] - test_data_slice.iloc[24*14:]['y_pred_tft'])** 2).mean() **.5

plt.legend(['Ground Truth','xgboost RMSE: {}'.format(round(rmse_xgb,2)),'TFT RMSE: {}'.format(round(rmse_tft,2))])
plt.xlim(test_data_slice.index[24*14], )
plt.title('Model Comparison')


# ### Variable weightings:
# 
# The variable selection network uses a learned sparse weight matrix to weight the input variable embeddings. We show the weights 

# In[ ]:


pd.DataFrame(encoder_sparse_weights[:,:,0,:].mean(axis=1).mean(axis=0).detach().unsqueeze(0).cpu().numpy(), columns=all_cols[:]+['time_index']).T.sort_values(0).plot(kind='barh')

plt.xlabel('Encoder Variable Weights')


# In[ ]:


pd.DataFrame(decoder_sparse_weights[:,:,0,:].mean(axis=1).mean(axis=0).detach().unsqueeze(0).cpu().numpy(), columns=all_cols[1:]+['time_index']).T.sort_values(0).plot(kind='barh')
plt.xlabel('Decoder Variable Weights')

