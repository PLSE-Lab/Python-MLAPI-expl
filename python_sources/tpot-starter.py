#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd
from tpot import TPOTRegressor


# In[ ]:


train = pd.read_csv('../input/train.csv')
X = train[train.columns[2:]].values
y = train['target'].values


# In[ ]:


X_train, y_train = X, y


# In[ ]:


tpot = TPOTRegressor(generations=8, population_size=10, verbosity=2)
tpot = tpot.fit(X_train, np.log1p(y_train))


# In[ ]:


test = pd.read_csv('../input/test.csv')
X_t = test[test.columns[1:]].values
y_pred = tpot.predict(X_t)


# In[ ]:


sub = pd.read_csv('../input/sample_submission.csv')
sub['target'] = np.around(np.expm1(y_pred), 0)
sub.to_csv('sub_tpot.csv', index=False)

