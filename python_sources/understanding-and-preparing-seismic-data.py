#!/usr/bin/env python
# coding: utf-8

# # General information
# Forecasting earthquakes is one of the most important problems in Earth science because of their devastating consequences. Current scientific studies related to earthquake forecasting focus on three key points: **when** the event will occur, **where** it will occur, and **how** large it will be. In this competition we try to predict time left to the next laboratory earthquake based on seismic signal data to answer the question of **when** earthquake will occur.
# 
# Training data represents one huge signal, but in test data we have many separate chunks, for each of which we need to predict time to failure.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# The data is huge, training data contains nearly **600 million rows** and that is A LOT of data to understand. I am just curious to know how long Kaggle servers will take to read this data.

# In[ ]:


get_ipython().run_cell_magic('time', '', "train = pd.read_csv('../input/train.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})")


# In[ ]:


train.head(5)


# In[ ]:


train.shape


# In[ ]:


partial_train = train[::20]


# In[ ]:


partial_train.head(5)


# In[ ]:


partial_train['time_to_failure'].shape


# In[ ]:


figure, axes1 = plt.subplots(figsize=(18,10))

plt.title("Seismic Data Trends with 5% sample of original data")

plt.plot(partial_train['acoustic_data'], color='r')
axes1.set_ylabel('Acoustic Data', color='r')
plt.legend(['Acoustic Data'])

axes2 = axes1.twinx()
plt.plot(partial_train['time_to_failure'], color='g')
axes2.set_ylabel('Time to Failure', color='g')
plt.legend(['Time to Failure'])


# ## Summary
# Data is **HUGE!** <br>
# Before every Failure, **there is a peak** in Acoustic Data. <br>
# Since we only have acoustic data to predict the time to failure, we need to **generate some features**. <br>
# I would use **Feature Engineering** to generate some common statistical features like **Mean, Variance, Max, Min, Std. Dev.** of our acoustic data. <br>
# Since the test data is **segmented into chunks of data**, it is better to segment our training data into chunks and then generate the features. <br>
# 

# # Data Preparation
# It took about 6 million datapoints to get one failure. Each of test csv files contains 150,000 datapoints. Thus, we should be able to get about **629145480 / 150000 = 4194** samples (subsets similar to our test dataset for training our model) before the first failure in our training set. Their corresponding labels should be the time_to_failure at the last datapoint in each subset. We would use a regression model to predict the time to failure for each corresponding subset sample.

# In[ ]:


# list of features to be engineered

features = ['mean','max','variance','min', 'stdev', 'max-min-diff',
            'max-mean-diff', 'mean-change-abs', 'abs-max', 'abs-min',
            'std-first-50000', 'std-last-50000', 'mean-first-50000',
            'mean-last-50000', 'max-first-50000', 'max-last-50000',
            'min-first-50000', 'min-last-50000']


# In[ ]:


# Feature Engineering

rows = 150000
segments = int(np.floor(train.shape[0] / rows))

X = pd.DataFrame(index=range(segments), dtype=np.float64,
                       columns=features)
Y = pd.DataFrame(index=range(segments), dtype=np.float64,
                       columns=['time_to_failure'])

for segment in range(segments):
    seg = train.iloc[segment*rows:segment*rows+rows]
    x = seg['acoustic_data'].values
    y = seg['time_to_failure'].values[-1]
    
    Y.loc[segment, 'time_to_failure'] = y
    
    X.loc[segment, 'mean'] = x.mean()
    X.loc[segment, 'stdev'] = x.std()
    X.loc[segment, 'variance'] = np.var(x)
    X.loc[segment, 'max'] = x.max()
    X.loc[segment, 'min'] = x.min()
    X.loc[segment, 'max-min-diff'] = x.max()-x.min()
    X.loc[segment, 'max-mean-diff'] = x.max()-x.mean()
    X.loc[segment, 'mean-change-abs'] = np.mean(np.diff(x))
    X.loc[segment, 'abs-min'] = np.abs(x).min()
    X.loc[segment, 'abs-max'] = np.abs(x).max()
    X.loc[segment, 'std-first-50000'] = x[:50000].std()
    X.loc[segment, 'std-last-50000'] = x[-50000:].std()
    X.loc[segment, 'mean-first-50000'] = x[:50000].min()
    X.loc[segment, 'mean-last-50000'] = x[-50000:].mean()
    X.loc[segment, 'max-first-50000'] = x[:50000].max()
    X.loc[segment, 'max-last-50000'] = x[-50000:].max()
    X.loc[segment, 'min-first-50000'] = x[:50000].min()
    X.loc[segment, 'min-last-50000'] = x[-50000:].min()


# In[ ]:


X.head(5)


# In[ ]:


data = pd.concat([X,Y],axis=1)


# In[ ]:


sns.set(rc={'figure.figsize': (18,12)})
sns.pairplot(data)


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler


# In[ ]:


X_train,X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state=1210)


# In[ ]:


scaler = StandardScaler()
scaler.fit(X_train)
X_train_sc = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)

scaler.fit(X_test)
X_test_sc = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)


# In[ ]:


lr = LinearRegression()


# In[ ]:


lr.fit(X_train_sc,y_train)


# In[ ]:


pred = lr.predict(X_test_sc)


# In[ ]:


mean_absolute_error(y_test, pred)


# In[ ]:


from lightgbm import LGBMRegressor
params = {'num_leaves': 54,'min_data_in_leaf': 79,'objective': 'huber',
         'max_depth': -1, 'learning_rate': 0.01, "boosting": "gbdt",
         # "feature_fraction": 0.8354507676881442,
         "bagging_freq": 3,"bagging_fraction": 0.8126672064208567,
         "bagging_seed": 11,"metric": 'mae',
         "verbosity": -1,'reg_alpha': 1.1302650970728192,
         'reg_lambda': 0.3603427518866501}


# In[ ]:


lgbm = LGBMRegressor(nthread=4,n_estimators=10000,
            learning_rate=0.01,num_leaves=54,
            colsample_bytree=0.9497036,subsample=0.8715623,
            max_depth=8,reg_alpha=0.04,
            reg_lambda=0.073,min_child_weight=40,silent=-1,verbose=-1,)


# In[ ]:


lgbm.fit(X_train, y_train)


# In[ ]:


pred_lgbm = lgbm.predict(X_test)


# In[ ]:


mean_absolute_error(y_test,pred_lgbm)


# The score can definitely be improved a lot. I will come back with more insights and update this kernel for better scores. Maybe using multiple predictors and the averaging their results will help reduce the error.
