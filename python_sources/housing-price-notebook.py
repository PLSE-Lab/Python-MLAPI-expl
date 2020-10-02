#!/usr/bin/env python
# coding: utf-8

# This my first notebook on Kaggle.  I will be working on Russian Housing price prediction problem. There are many independent variables in the data. I will use ML algorithms to find the best features

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection, preprocessing
import xgboost as xgb
color = sns.color_palette()

get_ipython().run_line_magic('matplotlib', 'inline')

pd.options.mode.chained_assignment = None  # default='warn'
pd.set_option('display.max_columns', 600)


# Reading the datasets

# In[ ]:


train_df = pd.read_csv("../input/train.csv")
train_df.shape


# Observing the variables by seeing header

# In[ ]:


train_df.head()
train_df.tail() 


# to get a list of the column headers from a pandas DataFrame

# In[ ]:


plt.figure(figsize=(10,8))
plt.scatter(range(train_df.shape[0]), np.sort(train_df.price_doc.values))
plt.xlabel('obs', fontsize=12)
plt.ylabel('price', fontsize=12)
plt.show()


# Checking the data types of the variables in train_df

# In[ ]:


train_df.dtypes
train_df.columns


# In[ ]:


for f in train_df.columns:
    if train_df[f].dtype=='object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train_df[f].values)) 
      
        train_df[f] = lbl.transform(list(train_df[f].values))


# In[ ]:


train_df.dtypes


# In[ ]:


for f in train_df.columns:
    if train_df[f].dtype=='object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train_df[f].values)) 
        train_df[f] = lbl.transform(list(train_df[f].values))
        
train_y = train_df.price_doc.values
train_X = train_df.drop(["id", "timestamp", "price_doc"], axis=1)

xgb_params = {
    'eta': 0.05,
    'max_depth': 8,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}
dtrain = xgb.DMatrix(train_X, train_y, feature_names=train_X.columns.values)
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=100)

# plot the important features #
fig, ax = plt.subplots(figsize=(12,18))
xgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
plt.show()


# Label encoding of categories 

# In[ ]:


train_y = train_df.price_doc.values
train_y
train_X = train_df.drop(["id", "timestamp", "price_doc"], axis=1)
train_X.head()


# In[ ]:


for f in train_df.columns:
    if train_df[f].dtype=='object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train_df[f].values)) 
        train_df[f] = lbl.transform(list(train_df[f].values))
import math
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestRegressor
n_features=40
train_df=train_df.fillna(train_df.median())  
target=train_df['price_doc']
rf = RandomForestRegressor(n_estimators=20,max_features=int(math.sqrt(n_features)),random_state=0)
rf.fit(train_df, target)
row=list(train_df.columns.values)
feature_names = np.array(row)
importances1 = rf.feature_importances_
important_names1 = feature_names[importances1 > np.mean(importances1)]
important_names1

