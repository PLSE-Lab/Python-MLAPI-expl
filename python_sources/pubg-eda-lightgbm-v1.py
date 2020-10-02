#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8
"""
@author: nakayama.s
"""

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


# In[ ]:


#!/usr/bin/env python
# coding: utf-8
"""
@author: nakayama.s
"""

import os
import warnings
import gc
import time
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.metrics import log_loss
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from itertools import product
from sklearn.model_selection import train_test_split
import lightgbm as lgb


# In[ ]:


def graph_insight(data):
    print(set(data.dtypes.tolist()))
    df_num = data.select_dtypes(include = ['float64', 'int64'])
    df_num.hist(figsize=(16, 16), bins=50, xlabelsize=8, ylabelsize=8);

def eda(data):
    # print(data)
    print("----------Top-5- Record----------")
    print(data.head(5))
    print("-----------Information-----------")
    print(data.info())
    print("-----------Data Types------------")
    print(data.dtypes)
    print("----------Missing value----------")
    print(data.isnull().sum())
    print("----------Null value-------------")
    print(data.isna().sum())
    print("----------Shape of Data----------")
    print(data.shape)
    print("----------describe---------------")
    print(data.describe())
    print("----------tail-------------------")
    print(data.tail())
    
def read_csv(path):
  # logger.debug('enter')
  df = pd.read_csv(path)
  # logger.debug('exit')
  return df

def load_train_data():
  # logger.debug('enter')
  df = read_csv(SALES_TRAIN_V2)
  # logger.debug('exit')
  return df

def load_test_data():
  # logger.debug('enter')
  df = read_csv(TEST_DATA)
  # logger.debug('exit')
  return df

def graph_insight(data):
    print(set(data.dtypes.tolist()))
    df_num = data.select_dtypes(include = ['float64', 'int64'])
    df_num.hist(figsize=(16, 16), bins=50, xlabelsize=8, ylabelsize=8);

def drop_duplicate(data, subset):
    print('Before drop shape:', data.shape)
    before = data.shape[0]
    data.drop_duplicates(subset,keep='first', inplace=True) #subset is list where you have to put all column for duplicate check
    data.reset_index(drop=True, inplace=True)
    print('After drop shape:', data.shape)
    after = data.shape[0]
    print('Total Duplicate:', before-after)

def unresanable_data(data):
    print("Min Value:",data.min())
    print("Max Value:",data.max())
    print("Average Value:",data.mean())
    print("Center Point of Data:",data.median())

SAMPLE_SUBMISSION    = '../input/sample_submission_V2.csv'
TRAIN_DATA           = '../input/train_V2.csv'
TEST_DATA            = '../input/test_V2.csv'

sample       = read_csv(SAMPLE_SUBMISSION)
train           = read_csv(TRAIN_DATA)
test            = read_csv(TEST_DATA)


# In[ ]:


eda(train)


# In[ ]:


eda(test)


# In[ ]:


train = pd.get_dummies(train,columns=['matchType'])


# In[ ]:


train =train.dropna()


# In[ ]:


test = pd.get_dummies(test,columns=['matchType'])


# In[ ]:


y_train =train['winPlacePerc']
x_train =train.drop(['Id','groupId','matchId','winPlacePerc'],axis=1)


# In[ ]:


X_train = x_train.values
Y_train = y_train.values


# In[ ]:


X_train.shape


# In[ ]:


Y_train.shape


# In[ ]:


# validation_size = 0.20
# seed = 1000
# X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, test_size=validation_size, random_state=seed)


# In[ ]:


# XGBoost
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

print("Parameter optimization")
xgb_model = xgb.XGBRegressor()
reg_xgb = GridSearchCV(xgb_model,
                   {'max_depth': [1,3,5],
                    'n_estimators': [50,100,200]},cv=5 ,verbose=1)
reg_xgb.fit(x_train, y_train)


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor

def create_model(optimizer='adam'):
    model = Sequential()
    model.add(Dense(X_train.shape[1], input_dim=X_train.shape[1], kernel_initializer='normal', activation='relu'))
    model.add(Dense(16, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))

    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

model = KerasRegressor(build_fn=create_model, verbose=0)
# define the grid search parameters
optimizer = ['SGD','Adam']
batch_size = [10, 30, 50]
epochs = [10, 50, 100]
param_grid = dict(optimizer=optimizer, batch_size=batch_size, epochs=epochs)
reg_dl = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
reg_dl.fit(X_train, y_train)


# In[ ]:


train_data = lgb.Dataset(data=X_train, label=Y_train)
valid_data = lgb.Dataset(data=X_valid, label=Y_valid)

params = {
    'num_leaves': 144,
    "metric" : "mae",
    'learning_rate': 0.1,
    'n_estimators': 800,
    'max_depth':13,
    'max_bin':55,
    'bagging_fraction':0.8,
    'bagging_freq':5,
    'feature_fraction':0.9
    }

lgb_model = lgb.train(params, train_data, valid_sets=[train_data, valid_data], verbose_eval=1000)


# In[ ]:


X_test = test.drop(["Id", "groupId", "matchId"], axis = 1)


# In[ ]:





# In[ ]:


Y_test = lgb_model.predict(X_test)

submission = pd.DataFrame({
    "Id": test['Id'],
    "winPlacePerc": Y_test
})

submission.to_csv('sample_submission_20181124_v1.csv', index=False)


# In[ ]:


submission.head()

