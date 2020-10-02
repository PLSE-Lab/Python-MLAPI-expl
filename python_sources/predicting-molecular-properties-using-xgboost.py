#!/usr/bin/env python
# coding: utf-8

# # Predicting Molecular Properties

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, GroupKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn import metrics
import lightgbm as lgb
from xgboost import XGBRegressor
# Input data files are available in the "../input/" directory.


# ## Loading the data

# In[ ]:


train = pd.read_csv('../input/champs-scalar-coupling/train.csv')
test = pd.read_csv('../input/champs-scalar-coupling/test.csv')
structures = pd.read_csv('../input/champs-scalar-coupling/structures.csv')


# In[ ]:


print('Train shape: {}'.format(train.shape))
print('Test shape: {}'.format(test.shape))


# ## EDA

# In[ ]:


train.info()


# In[ ]:


test.info()


# In[ ]:


train.describe()


# In[ ]:


train[['molecule_name','scalar_coupling_constant']].groupby('molecule_name').mean()[:100]


# In[ ]:


train[['type','scalar_coupling_constant']].groupby('type').count()


# In[ ]:


sns.distplot(train['scalar_coupling_constant'])


# > 

# ## Feature Engineering

# In[ ]:


train = pd.merge(train, structures, how = 'left',left_on  = ['molecule_name', 'atom_index_0'],
right_on = ['molecule_name',  'atom_index'])

train = pd.merge(train, structures, how = 'left',left_on  = ['molecule_name', 'atom_index_1'],
right_on = ['molecule_name',  'atom_index'])


# In[ ]:


test = pd.merge(test, structures, how = 'left',left_on  = ['molecule_name', 'atom_index_0'],
right_on = ['molecule_name',  'atom_index'])

test = pd.merge(test, structures, how = 'left',left_on  = ['molecule_name', 'atom_index_1'],
right_on = ['molecule_name',  'atom_index'])


# In[ ]:


train.head()


# In[ ]:


train['dist'] = ((train['x_y'] - train['x_x'])**2 + (train['y_y'] - train['y_x'])**2 + 
(train['z_y'] - train['z_x'])**2 ) ** 0.5

test['dist'] = ((test['x_y'] - test['x_x'])**2 + (test['y_y'] - test['y_x'])**2 +
(test['z_y'] - test['z_x'])**2 ) ** 0.5


# In[ ]:


train.head()


# > ## Modeling

# In[ ]:


train.columns


# In[ ]:


features = ['atom_index_x', 'x_x', 'y_x','z_x', 'atom_index_y', 'x_y', 'y_y', 'z_y', 'dist']


# In[ ]:


X_train,X_val,y_train,y_val = train_test_split(train[features], train['scalar_coupling_constant'],test_size=0.2)


# In[ ]:


xgb = XGBRegressor()
xgb.fit(X_train,y_train)
preds = xgb.predict(X_val)


# In[ ]:


np.log(metrics.mean_absolute_error(y_val,preds))


# In[ ]:


test_predictions = xgb.predict(test[features])


# In[ ]:


sns.distplot(test_predictions)


# Looks like the distribution of the scalar coupling constant in the train dataset.

# In[ ]:


submission = pd.DataFrame()
submission['id'] = test['id']
submission['scalar_coupling_constant'] = test_predictions


# In[ ]:


submission.to_csv('usingXGBoost.csv.gz',index=False,compression='gzip')


# This is my first competition and first kernel on Kaggle. If you think I've made some mistakes or I can improve somewhere, please let me know. Thanks :)

# In[ ]:




