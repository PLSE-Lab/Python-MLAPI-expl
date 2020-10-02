#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import pickle
import os
print(os.listdir("../input/feature-engineering/"))
print(os.listdir("../input/elo-merchant-category-recommendation/"))
import datetime
import time
import sys
# Any results you write to the current directory are saved as output.


# In[ ]:


with open('../input/feature-engineering/new_sum.pickle', 'rb') as f:
    new_sum = pickle.load(f)
with open('../input/feature-engineering/his_sum.pickle', 'rb') as f:
    his_sum = pickle.load(f)


# In[ ]:


train = pd.read_csv( '../input/elo-merchant-category-recommendation/train.csv',parse_dates =["first_active_month"])


# In[ ]:


new_sum.head()
new_sum.columns.values[1] = 'new_card_id_size'
new_sum.columns


# In[ ]:


his_sum.head()
his_sum.columns.values[1] = 'his_card_id_size'
his_sum.columns


# In[ ]:


train_his=train.merge(his_sum, how='right', on="card_id")


# In[ ]:


train_his.head()
train_his.shape


# In[ ]:


train_his_new=train_his.merge(new_sum, how='right', on="card_id")


# In[ ]:


train_his_new.shape


# In[ ]:


train_his_new.head()


# In[ ]:


train_his_new.columns.values


# In[ ]:


train_his_new['first_active_month'] = pd.to_datetime(train_his_new['first_active_month'])
train_his_new['first_active_years'] = train_his_new['first_active_month'].dt.year
train_his_new['first_active_months'] = train_his_new['first_active_month'].dt.month
train_his_new['howlong'] = (datetime.date(2018,2,1) - train_his_new['first_active_month'].dt.date).dt.days


# In[ ]:


train_his_new.shape


# In[ ]:


train_matrix = train_his_new.drop(['card_id','first_active_month','target'], axis=1)


# In[ ]:


train_matrix.shape


# In[ ]:


target = train_his_new['target']


# # (?)Feature Selection Using Random forest
# https://towardsdatascience.com/feature-selection-using-random-forest-26d7b747597f

# In[ ]:


# from sklearn.ensemble import RandomForestRegressor
# from sklearn.feature_selection import SelectFromModel


# In[ ]:


# sel = SelectFromModel(RandomForestRegressor(n_estimators = 100))
# sel.fit(train_matrix, target)


# In[ ]:


# train_matrix[train_matrix.isnull().any(axis=1)]


# # A Feature Selection Tool for Machine Learning in Python
# https://towardsdatascience.com/a-feature-selection-tool-for-machine-learning-in-python-b64dd23710f0

# In[ ]:


import sys
sys.path.append('../input/feature-selector/')
from feature_selector import FeatureSelector
# Features are in train and labels are in train_labels
fs = FeatureSelector(data = train_matrix, labels = target)


# In[ ]:


fs.identify_missing(missing_threshold = 0.6)


# In[ ]:


fs.missing_stats.head()


# In[ ]:


fs.plot_missing()


# In[ ]:


fs.identify_collinear(correlation_threshold = 0.98)


# In[ ]:


fs.plot_collinear()


# In[ ]:


# list of collinear features to remove
collinear_features = fs.ops['collinear']
# dataframe of collinear features
fs.record_collinear.head()


# In[ ]:


# Pass in the appropriate parameters
fs.identify_zero_importance(task = 'regression', 
                            eval_metric = 'auc', 
                            n_iterations = 10, 
                             early_stopping = True)
# list of zero importance features
zero_importance_features = fs.ops['zero_importance']


# In[ ]:


# plot the feature importances
fs.plot_feature_importances(threshold = 0.99, plot_n = 12)


# In[ ]:


fs.identify_low_importance(cumulative_importance = 0.99)


# In[ ]:


fs.identify_single_unique()


# In[ ]:


# Remove the features from all methods (returns a df)
train_removed = fs.remove(methods = 'all')


# In[ ]:


train_removed.shape
with open('train_removed.pickle', 'wb') as f:
    pickle.dump(train_removed, f)
with open('target.pickle', 'wb') as f:
    pickle.dump(target, f)


# In[ ]:


#train_removed[train_removed.isnull().any(axis=1)]


# In[ ]:




