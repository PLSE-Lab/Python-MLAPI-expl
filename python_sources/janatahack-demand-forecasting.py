#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, RepeatedStratifiedKFold, cross_val_score, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import StandardScaler, Normalizer, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from xgboost import XGBRegressor

from catboost import CatBoostRegressor
from catboost import Pool

import datetime
from datetime import datetime


# In[ ]:


def parser(x):
    return datetime.strptime(x, '%d/%m/%y')


# In[ ]:


train = pd.read_csv('/kaggle/input/train.csv', parse_dates = [1], date_parser = parser)
train = train.drop('record_ID', axis = 1)
test = pd.read_csv('/kaggle/input/test.csv', parse_dates = [1], date_parser = parser)
test_ID = test.record_ID
test = test.drop('record_ID', axis = 1)

train.loc[(train['total_price'].isnull()), 'total_price'] = train.loc[(train['total_price'].isnull()), 'base_price'].values[0].astype(float)

train['quarter'] = train['week'].dt.quarter
train['year'] = train['week'].dt.year
train['month'] = train['week'].dt.month
train['date'] = train['week'].dt.day
train['week_day'] = train['week'].dt.dayofweek
train['is_weekend'] = np.where(train['week'].isin([5, 6]), 1, 0)
train['is_weekday'] = np.where(train['week'].isin([0, 1, 2, 3, 4]), 1, 0)

test['quarter'] = test['week'].dt.quarter
test['year'] = test['week'].dt.year
test['month'] = test['week'].dt.month
test['date'] = test['week'].dt.day
test['week_day'] = test['week'].dt.dayofweek
test['is_weekend'] = np.where(test['week'].isin([5, 6]), 1, 0)
test['is_weekday'] = np.where(test['week'].isin([0, 1, 2, 3, 4]), 1, 0)


# In[ ]:


train.head()


# In[ ]:


train.info()


# In[ ]:


test.info()


# In[ ]:


train.head()


# In[ ]:


plt.figure(figsize = (16, 7))
sns.lineplot(train['week'], train['units_sold'])


# In[ ]:


plt.figure(figsize = (16, 7))
sns.barplot(train['store_id'], train['units_sold'])
plt.xticks(rotation = 90)
plt.show()


# In[ ]:


train.loc[(train['total_price'].isnull()), 'total_price'] = train.loc[(train['total_price'].isnull()), 'base_price'].values[0].astype(float)
plt.figure(figsize = (16, 7))
sns.lineplot(train['total_price'], train['units_sold'])
sns.despine()


# In[ ]:


plt.figure(figsize = (16, 7))
sns.barplot(train['sku_id'], train['units_sold'])
plt.xticks(rotation = '90')
plt.show()


# In[ ]:


plt.figure(figsize = (16, 7))
sns.lineplot(train['store_id'], train['units_sold'])
plt.xticks(rotation = '90')


# In[ ]:


plt.figure(figsize = (16, 7))
sns.boxplot(x = 'units_sold', data = train)
sns.despine()


# In[ ]:


plt.figure(figsize = (16, 7))
sns.jointplot(x = 'total_price', y = 'units_sold', data = train)
plt.show()


# In[ ]:


plt.figure(figsize = (16, 7))
sns.jointplot(x = 'base_price', y = 'units_sold', data = train)
plt.show()


# In[ ]:


plt.figure(figsize = (16, 7))
sns.jointplot(x = 'store_id', y = 'units_sold', data = train)


# In[ ]:


plt.figure(figsize = (16, 7))
sns.boxplot(x = 'sku_id', y = 'units_sold', data = train)
plt.xticks(rotation = 90)
plt.show()


# In[ ]:


print("Total number of stores : ", train['store_id'].nunique())


# In[ ]:


store_sku_train = (train['store_id'].astype(str) + "_" + train['sku_id'].astype(str)).unique()
print("There are", len(store_sku_train), "store-product pairs in train data")


# In[ ]:


store_sku_test = (test['store_id'].astype(str) + "_" + test['sku_id'].astype(str)).unique()
print("There are", len(store_sku_test), "store-product pairs in test data")


# In[ ]:


# check if test set has any new center-mean pair or not
print("There are",len(set(store_sku_test) - set(store_sku_train)), "New center-meal pairs in test dataset which are not present in train dataset")
print(set(store_sku_test) - set(store_sku_train))


# #### Remove outlier

# In[ ]:


outlier_index = train[(train['units_sold'] > 1500)].index
train.drop(outlier_index, inplace = True)


# #### Feature Engineering

# In[ ]:


plt.figure(figsize = (12, 10))
sns.heatmap(train.corr())
sns.despine


# In[ ]:


avg_sku_bp = train[['sku_id', 'base_price']].append(test[['sku_id', 'base_price']])
avg_sku_bp.columns = ['sku_id', 'avg_bp']
avg_sku_bp = avg_sku_bp.groupby(['sku_id'])['avg_bp'].mean()

train = pd.merge(train, avg_sku_bp, on = ["sku_id"], how = "left")
train['bp_fraction'] = train['base_price']/train['avg_bp']

test = pd.merge(test, avg_sku_bp, on = ["sku_id"], how = "left")
test['bp_fraction'] = test['base_price']/test['avg_bp']


# In[ ]:


train['discount'] = train['base_price'] - train['total_price']
train['discount_per'] = (train['discount']/train['base_price'])*100
train['promo_homepage'] = train['is_display_sku'] + train['is_featured_sku']
train['store_id'] = train['store_id'].astype(np.object)
train['sku_id'] = train['sku_id'].astype(np.object)

train = pd.get_dummies(train, drop_first = True)
train = train.drop('week', axis = 1)

test['discount'] = test['base_price'] - test['total_price']
test['discount_per'] = (test['discount']/test['base_price'])*100
test['promo_homepage'] = test['is_display_sku'] + test['is_featured_sku']
test['store_id'] = test['store_id'].astype(np.object)
test['sku_id'] = test['sku_id'].astype(np.object)

test = pd.get_dummies(test, drop_first = True)
test = test.drop('week', axis = 1)


# In[ ]:


train.head()


# In[ ]:


X, y = train.drop(['units_sold'],axis = 1), train.units_sold
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 22)


# In[ ]:


rf = RandomForestRegressor(n_estimators = 200)
rf.fit(X_train, y_train)
test_pred = rf.predict(X_test)
print(100*np.sqrt(mean_squared_log_error(y_test, test_pred)))

print('RMSLE score is', 100*(np.sqrt(np.mean(np.power(np.log1p(y_test)-np.log1p(test_pred), 2)))))


# In[ ]:


sorted(zip(rf.feature_importances_, X_train), reverse = True)


# In[ ]:


y_pred = rf.predict(test)


# In[ ]:


submission = pd.read_csv('/kaggle/input/sample_submission.csv')
submission.head()


# In[ ]:


submission = pd.DataFrame({'record_ID': test_ID, 'units_sold': y_pred})
submission.to_csv('RandomForest_final.csv',index = False)
submission.head()


# #### AV Score: 475.75, Rank: 123  (Jupyter Notebook: 42.80849288634156 - without Quarter and Avg. Base Price,  fraction) - RandomForest_v2.csv
