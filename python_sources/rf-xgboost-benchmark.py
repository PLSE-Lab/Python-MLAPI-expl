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

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# # Input Data

# In[ ]:


train = pd.read_csv('../input/train.csv', parse_dates=['date'])
test = pd.read_csv('../input/test.csv', parse_dates=['date'])


# In[ ]:


print("Train shape: ", train.shape)
print("Test shape: ", test.shape)


# In[ ]:


train.describe()


# # Date Preprocess

# In[ ]:


df = pd.concat([train,test])
df['month'] = df['date'].dt.month
df['weekday'] = df['date'].dt.dayofweek
df['year'] = df['date'].dt.year
df['week_of_year']  = train.date.dt.weekofyear

df.drop('date', axis=1, inplace=True)
df.head()


# # Benchmark Model

# In[ ]:


col = [i for i in df.columns if i not in ['date','id','sales']]
y = 'sales'
train = df.loc[~df.sales.isna()]


# In[ ]:


from sklearn import cross_validation
X_train, X_test, y_train, y_test = cross_validation.train_test_split(train[col],train[y], test_size=0.2, random_state=42)
print(X_train.shape)
print(X_test.shape)
X_train.head()


# ## Linear Regression for benchmark model

# In[ ]:


from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit(X_train, y_train)


# In[ ]:


from sklearn.metrics import mean_absolute_error
predictions = reg.predict(X_test)
print(mean_absolute_error(predictions,y_test))


# # Correlations Study

# In[ ]:


import seaborn as sns
train2 = train.copy()
train2.drop('id', axis=1, inplace=True)
train2.head()
corr = train2.corr()
sns.heatmap(corr, xticklabels=corr.columns,yticklabels=corr.columns)


# All of feature has low correlation of sales. The original feature is not good for model training.

# # Feature Extraction
# Add historical / seasonal features. Thanks to [Dan Ofer's notebook](https://www.kaggle.com/danofer/getting-started-with-time-series-features)

# In[ ]:


df["median-store_item-month"] = df.groupby(['month',"item","store"])["sales"].transform("median")
df["mean-store_item-week"] = df.groupby(['week_of_year',"item","store"])["sales"].transform("mean")
df["item-month-sum"] = df.groupby(['month',"item"])["sales"].transform("sum") # total sales of that item  for all stores
df["store-month-sum"] = df.groupby(['month',"store"])["sales"].transform("sum") 


# In[ ]:


df.head()


# In[ ]:


train = df.loc[~df.sales.isna()]
train3 = train.copy()
train3.drop('id', axis=1, inplace=True)
train3.head()
corr = train3.corr()
sns.heatmap(corr, xticklabels=corr.columns,yticklabels=corr.columns)


# In[ ]:


# get shifted features for grouped data. Note need to sort first! 
df["item-week_shifted-90"] = df.groupby(['week_of_year',"item"])["sales"].transform(lambda x:x.shift(12).sum()) # shifted total sales for that item 12 weeks (3 months) ago
df["store-week_shifted-90"] = df.groupby(['week_of_year',"store"])["sales"].transform(lambda x:x.shift(12).sum()) # shifted total sales for that store 12 weeks (3 months) ago
df["item-week_shifted-90"] = df.groupby(['week_of_year',"item"])["sales"].transform(lambda x:x.shift(12).mean()) # shifted mean sales for that item 12 weeks (3 months) ago
df["store-week_shifted-90"] = df.groupby(['week_of_year',"store"])["sales"].transform(lambda x:x.shift(12).mean()) # shifted mean sales for that store 12 weeks (3 months) ago


# In[ ]:


train = df.loc[~df.sales.isna()]
train4 = train.copy()
train4.drop('id', axis=1, inplace=True)
train4.head()
corr = train4.corr()
sns.heatmap(corr, xticklabels=corr.columns,yticklabels=corr.columns)


# The correlation looks better after feature extraction.

# ## Miss data review

# In[ ]:


import missingno as msno
msno.bar(train,figsize=(20,4))


# # Final Model

# In[ ]:


col = [i for i in train.columns if i not in ['id','sales','store','item','month','weekday','year','week_of_year']]
y = 'sales'
from sklearn import cross_validation
X_train, X_test, y_train, y_test = cross_validation.train_test_split(train[col],train[y], test_size=0.2, random_state=42)
print(X_train.shape)
print(X_test.shape)
X_train.head()


# ## Linear Regression

# In[ ]:


from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit(X_train, y_train)
from sklearn.metrics import mean_absolute_error
predictions = reg.predict(X_test)
print(mean_absolute_error(predictions,y_test))


# ## Random Forst

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
RF = RandomForestRegressor(n_estimators=200, n_jobs=-1)
RF.fit(X_train, y_train)
from sklearn.metrics import mean_absolute_error
predictions = RF.predict(X_test)
print(mean_absolute_error(predictions,y_test))


# ## XGBoost

# In[ ]:


import xgboost
xgb = xgboost.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,
                           colsample_bytree=1, max_depth=7)
xgb.fit(X_train,y_train)
from sklearn.metrics import mean_absolute_error
predictions = xgb.predict(X_test)
print(mean_absolute_error(predictions,y_test))


# No matter what model is, the final testing performance (mean absolute error) after feature extracion are batter than benchmark model 
