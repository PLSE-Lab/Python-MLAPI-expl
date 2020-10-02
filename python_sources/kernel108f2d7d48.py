#!/usr/bin/env python
# coding: utf-8

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


train = pd.read_csv('/kaggle/input/walmart-recruiting-store-sales-forecasting/train.csv.zip', parse_dates = ['Date'])
train['year'] = train['Date'].dt.year
train['month'] = train['Date'].dt.month
train['day'] = train['Date'].dt.day
train['week'] = train['Date'].dt.week
train['week_num'] = np.ceil(train['day'] / 7)
train.head()


# In[ ]:


stores = pd.read_csv('/kaggle/input/walmart-recruiting-store-sales-forecasting/stores.csv')
stores.head()


# In[ ]:


train['week_num'] = np.ceil(train['day'] / 7)


# In[ ]:


train = pd.merge(train, stores, on = 'Store', how = 'left')
train.head()


# In[ ]:


train = train.replace({'A':1, 'B':2, 'C':3})
train.head()


# In[ ]:


train.groupby('Type')['Weekly_Sales'].mean()


# In[ ]:


train3 = train[train['Weekly_Sales'] < 50000]


# In[ ]:


train.shape


# In[ ]:


train.groupby('day')['Weekly_Sales'].mean()


# In[ ]:


test = pd.read_csv('/kaggle/input/walmart-recruiting-store-sales-forecasting/test.csv.zip', parse_dates = ['Date'])
test['year'] = test['Date'].dt.year
test['month'] = test['Date'].dt.month
test['day'] = test['Date'].dt.day
test['week'] = test['Date'].dt.week
test['week_num'] = np.ceil(test['day'] / 7)
test.head()


# In[ ]:


test = pd.merge(test, stores, on = 'Store', how = 'left')


# In[ ]:


test = test.replace({'A':1, 'B':2, 'C':3})


# In[ ]:


train2 = train.drop(['Weekly_Sales','Date'],1)
train2.head()


# In[ ]:


test2 = test.drop('Date', 1)
test2.head()


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 100, n_jobs = 4, random_state = 1 )


# In[ ]:


rf.fit(train2, train['Weekly_Sales'])


# In[ ]:


result = rf.predict(test2)


# In[ ]:


sub = pd.read_csv('/kaggle/input/walmart-recruiting-store-sales-forecasting/sampleSubmission.csv.zip')
sub.head()


# In[ ]:


sub['Weekly_Sales'] = result
sub.head()


# In[ ]:


sub.to_csv('200120.csv', index = False)

