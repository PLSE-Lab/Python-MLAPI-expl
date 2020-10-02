#!/usr/bin/env python
# coding: utf-8

# # **COVID-19 LSTM Cases & Fatalities prediction**

# Exploratory analysis on COVID-19 cases followed by development of a model to predict cases and fatalities by region. The model used is a simple LSTM which takes in historical (t-13) cases and fatalities and then predicts both at time t. 

# **Imports**

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


import matplotlib.pyplot as plt
import seaborn as sns

from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import MinMaxScaler


# In[ ]:


raw_train_data = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/train.csv')
test_data = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/test.csv')


# In[ ]:


# change dtypes
train_data = raw_train_data
train_data['Date'] = pd.to_datetime(train_data['Date'])
train_data['Province_State'] = train_data['Province_State'].fillna('None')


# In[ ]:


# scale features independently
case_scaler = MinMaxScaler(feature_range=(0, 100))
fat_scaler = MinMaxScaler(feature_range=(0, 100))
train_data['ConfirmedCases_scaled'] = case_scaler.fit_transform(train_data['ConfirmedCases'].values.reshape(-1,1))
train_data['Fatalities_scaled'] = fat_scaler.fit_transform(train_data['Fatalities'].values.reshape(-1,1))


# # **EDA**

# Below are some temporal plots of the fatality and case data by region. The worst affected nations are shown

# In[ ]:


# plot global cases and fatalities temporally
global_cases = train_data.groupby(['Date'])[['ConfirmedCases','Fatalities']].sum()

fig,ax = plt.subplots(figsize=(8,5))
_=ax.plot(global_cases['ConfirmedCases'],label='Cases',c='k')
_=ax.plot(global_cases['Fatalities'],label='Deaths',c='r')
_=ax.xaxis.set_tick_params(rotation=15)
_=sns.despine()
_=ax.legend(loc=0)
_=ax.set(title=('Global cumulative cases and deaths'))


# In[ ]:


# highest infected countries as of 18/03/20
highest_cases_countries = (train_data[train_data['Date']=='2020-03-24']
                          .groupby(['Country_Region'])['ConfirmedCases']
                          .sum().sort_values(ascending=False)[:10])
print(highest_cases_countries)


# In[ ]:


all_regions = train_data.groupby(['Date','Country_Region'])[['ConfirmedCases','Fatalities']].sum()
# regions with cases and fatalities greater than 0
regions = all_regions[(all_regions['ConfirmedCases']>0)&
       (all_regions['Fatalities']>0)].reset_index()


# In[ ]:


fig,ax = plt.subplots(figsize=(8,5))
for i in highest_cases_countries.index:
    _=ax.plot(regions[regions['Country_Region']==i]['Date'],
             regions[regions['Country_Region']==i]['ConfirmedCases'],label=i)
    _=ax.legend()
    _=ax.xaxis.set_tick_params(rotation=15)
    _=sns.despine()
    _=ax.set(title='Regions with highest cases')


# # **Data prep**

# Prepare the data for input into the LSTM. The time series needs to be converted into a supervised learning problem with the correct dimensional inputs

# In[ ]:


# unique identifiers for each province and country
ids = (
    train_data[['Province_State','Country_Region']]
    .drop_duplicates().reset_index(drop=True).to_dict('index')
)


# In[ ]:


# dict of dfs for each province
region_dfs = {i:train_data[(train_data['Province_State']==ids[i]['Province_State'])&
          (train_data['Country_Region']==ids[i]['Country_Region'])] for i in ids}


# In[ ]:


# function to convert series into supervised learning problem
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
     # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# In[ ]:


# prepare data into sequences with n_in = lag time steps and n_out = future time steps
n_in = 13
n_out = 1
n_feats = 2 # cases and fatalities

regions_supervised = [series_to_supervised(region_dfs[i][['ConfirmedCases_scaled','Fatalities_scaled']],
                                           n_in=n_in, n_out=n_out, dropnan=True) for i in region_dfs]

reshaped_data = pd.concat(regions_supervised).reset_index(drop=True)


# In[ ]:


reshaped_data.shape


# In[ ]:


# # find sequences where cumulative deaths are 0 
# var2 = [col for col in reshaped_data.columns if 'var2' in col]
# zero_deaths_index = reshaped_data[var2][reshaped_data[var2].sum(axis=1)==0].index

# # remove 0 death sequences
# model_data = reshaped_data[~reshaped_data.index.isin(zero_deaths_index)]


# In[ ]:


# drop 5000 zero sequences
zero_indices = reshaped_data[reshaped_data.sum(axis=1)==0].sample(2000).index
model_data = reshaped_data.drop(zero_indices,axis=0)


# In[ ]:


model_data.shape


# In[ ]:


# LSTM input is 3D (samples, timesteps, feats), output is 2D(samples,feats)
# cross val data
restack = reshaped_data.values
x = restack[:,:-2].reshape(len(restack),n_in,2)
y = restack[:,-2:]


# In[ ]:


print(x.shape,y.shape)


# # **Model tuning**

# Build the model and tune the hyperparameters using grid search cross validation

# In[ ]:


def build_regressor(optimizer,lstm_nodes):
    
    # define model
    model = Sequential()
    model.add(LSTM(lstm_nodes, activation='relu', input_shape=(n_in,2)))
    model.add(Dense(2, activation='relu'))
    model.compile(optimizer=optimizer, loss='mean_squared_logarithmic_error')
    
    return model


# In[ ]:


# scikit wrapper function
regressor = KerasRegressor(build_fn = build_regressor,verbose=0)

# grid search parameters
parameters = {'lstm_nodes':[14,16,20],
             'nb_epoch':[50],
             'batch_size':[32],
             'optimizer':['adam']}


gridsearch = GridSearchCV(estimator = regressor,
                 param_grid = parameters,
                 scoring = 'neg_mean_squared_log_error',
                 cv = 10,
                 n_jobs = -1,
                 verbose =0)


# In[ ]:


# restack train and val 
gridsearch = gridsearch.fit(x,y)


# In[ ]:


# grid search results (evaluation metric in grid search cv is -ve mean squared log error, hence the manipulation here)
np.sqrt(-1*gridsearch.cv_results_['mean_test_score'])


# In[ ]:


gridsearch.cv_results_


# In[ ]:


gridsearch.best_params_


# # **Final model**

# Fit the final model using the best parameters from the grid search cross validation

# In[ ]:


# build final model with best params from grid search cv
final_regressor = build_regressor(optimizer=gridsearch.best_params_['optimizer'],
                                 lstm_nodes=gridsearch.best_params_['lstm_nodes'])


# In[ ]:


final_regressor.summary()


# In[ ]:


# fit final model
final_regressor.fit(x, y, 
                      epochs=gridsearch.best_params_['nb_epoch'],
                      batch_size=gridsearch.best_params_['batch_size'], 
                      verbose=0, 
                      shuffle=False)


# # **Predictions**

# The first days predictions are made using the prior 13 days data for each region. Then the modelled predictions are added to the dataset and fedback into the model to get the next days predictions, and so on for the entire test set. This means that model predictions are used as model inputs for later predictions.

# In[ ]:


test_data['Date'] = pd.to_datetime(test_data['Date'])
test_data.Date.max()-test_data.Date.min()


# In[ ]:


# first batch of predictions
first_predict_date = pd.to_datetime('2020-03-19')
pred_data = {key:region_dfs[key].loc[(region_dfs[key]['Date']>=(first_predict_date-pd.DateOffset(days=n_in)))&
                                    (region_dfs[key]['Date']<first_predict_date)] 
             for key in region_dfs}
test_reshaped = [pred_data[i][['ConfirmedCases_scaled','Fatalities_scaled']].values.reshape(1,n_in,2) for i in pred_data]
first_input = np.vstack(test_reshaped)
first_pred = final_regressor.predict(first_input)


# In[ ]:


# iterate prediction output back into model input for the next days
pin = [first_input]
pout = [first_pred]

# first prediction is done outside of loop, need to loop for following 41 days
for i in range(42):
    p = final_regressor.predict(pin[i])
    pout.append(p)
    t= np.insert(pin[i],n_in,pout[i],axis=1)[:,1:,:]
    pin.append(t)


# In[ ]:


# create the prediction dataframe
pred_df = pd.DataFrame(np.concatenate(pout))
pred_df.columns = ['ConfirmedCases_scaled','Fatalities_scaled']
pred_df['Date'] = np.repeat(test_data['Date'].unique(),294)
pred_df['Province_State'] = list(test_data.drop_duplicates(subset=['Province_State','Country_Region'])['Province_State'])*43
pred_df['Country_Region'] = list(test_data.drop_duplicates(subset=['Province_State','Country_Region'])['Country_Region'])*43

pred_df = pred_df.sort_values(by=['Country_Region','Date']).reset_index(drop=True)


# In[ ]:


# inverse scale results
pred_df['ConfirmedCases'] = (
    case_scaler.inverse_transform(pred_df['ConfirmedCases_scaled'].values.reshape(1,-1))[0])

pred_df['Fatalities'] = (
    fat_scaler.inverse_transform(pred_df['Fatalities_scaled'].values.reshape(1,-1))[0])


# In[ ]:


pred_df[pred_df['Date']=='2020-04-30'].sum()


# **Sanity check predictions**

# In[ ]:


raw_train_data['Date'] = pd.to_datetime(raw_train_data['Date'])
test_check = raw_train_data[raw_train_data['Date']>'2020-03-18']
model_check = pred_df[pred_df['Date']<=test_check['Date'].max()]


# In[ ]:


np.sqrt(mean_squared_log_error(y_true = test_check[['ConfirmedCases','Fatalities']],
                       y_pred = model_check[['ConfirmedCases','Fatalities']]))


# # Submission

# In[ ]:


sub = pred_df[['ConfirmedCases','Fatalities']]
sub['ForecastId'] = test_data['ForecastId']


# In[ ]:


sub.sample(20)


# In[ ]:


sub.to_csv("submission.csv",index=False)


# In[ ]:




