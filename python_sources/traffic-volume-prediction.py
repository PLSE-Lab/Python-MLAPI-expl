#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

from sklearn import tree
from sklearn import svm
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# **Load Data**

# In[ ]:


test_set = pd.read_csv("/kaggle/input/Test.csv")
train_set = pd.read_csv("/kaggle/input/Train.csv")


# In[ ]:


print(train_set.head())


# **Prepare Train Data**

# In[ ]:


train_set['date_time'] = pd.to_datetime(train_set.date_time)


# In[ ]:


train_set['year'] = train_set.date_time.dt.year
train_set['month'] = train_set.date_time.dt.month
train_set['day'] = train_set.date_time.dt.day
train_set['hour'] = train_set.date_time.dt.hour


# In[ ]:


train_copy = train_set.drop(['date_time'], axis=1)
print(train_copy.head())


# **One-hot-encoding**

# In[ ]:


train_onehot = train_copy.copy()


# In[ ]:


train_onehot = pd.get_dummies(train_onehot, columns=['is_holiday', 'weather_type', 'weather_description'], 
                              prefix=['is_holiday', 'weather_type', 'weather_desc'])


# In[ ]:


train_onehot = train_onehot.astype(float)
train_onehot.head()


# **Preparing the Test data**

# In[ ]:


test_set['date_time'] = pd.to_datetime(test_set.date_time)


# In[ ]:


test_set['year'] = test_set.date_time.dt.year
test_set['month'] = test_set.date_time.dt.month
test_set['day'] = test_set.date_time.dt.day
test_set['hour'] = test_set.date_time.dt.hour


# In[ ]:


test_copy = test_set.drop(['date_time'], axis=1)


# In[ ]:


test_onehot = test_copy.copy()
test_onehot = pd.get_dummies(test_onehot, columns=['is_holiday', 'weather_type', 'weather_description'], 
                              prefix=['is_holiday', 'weather_type', 'weather_desc'])


# In[ ]:


print(len(train_onehot.columns))
print(len(train_onehot))


# In[ ]:


print(len(test_onehot.columns))
print(len(test_onehot))


# In[ ]:


# find the features that are not in test data set
for x in train_onehot.columns:
    if x not in test_onehot.columns and x != 'traffic_volume':
        print(x)
        test_onehot[x] = train_onehot[x]


# In[ ]:


test_onehot = test_onehot.astype(float)


# **Train and Test data**

# In[ ]:


y_train = train_onehot['traffic_volume']
x_train = train_onehot.drop(['traffic_volume'], axis=1)


# **Training**

# In[ ]:


dec_tree_reg = tree.DecisionTreeRegressor()
dec_tree_reg.fit(x_train, y_train)


# **Decision Tree Regressor**

# In[ ]:


preds = dec_tree_reg.predict(test_onehot)
print(len(preds))
print(preds)
preds = preds.astype(int)
print(preds)


# In[ ]:


submission = pd.DataFrame(columns = ['date_time', 'traffic_volume']) 
print(submission.head())
submission.date_time = test_set.date_time
submission.traffic_volume = preds
print(len(submission))
print(submission.head())
submission.to_csv('dtreereg_final_prediction_submission.csv', index=False)


# In[ ]:




