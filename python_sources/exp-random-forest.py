#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.ensemble import RandomForestRegressor


# In[ ]:


train = pd.read_csv('../input/train.csv', index_col='Id')
test = pd.read_csv('../input/test.csv', index_col='Id')


# In[ ]:


target = train['SalePrice']  #target variable
train = train.drop('SalePrice', axis=1)
train['training_set'] = True
test['training_set'] = False


# In[ ]:


full = pd.concat([train, test])
full = full.interpolate()
full = pd.get_dummies(full)


# In[ ]:


train = full[full['training_set']==True]
train = train.drop('training_set', axis=1)
test = full[full['training_set']==False]
test = test.drop('training_set', axis=1)


# In[ ]:


rf = RandomForestRegressor(n_estimators=100, n_jobs=-1)
rf.fit(train, target)
preds = rf.predict(test)


# In[ ]:


my_submission = pd.DataFrame({'Id': test.index, 'SalePrice': preds})


# In[ ]:


my_submission.head(10000)


# In[ ]:


my_submission.to_csv('submission.csv',index=False)

