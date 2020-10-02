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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


dataset_train = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/train.csv')
dataset_test = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/test.csv')


# In[ ]:


dataset_train.head()


# In[ ]:


X_train = dataset_train.iloc[:, dataset_train.columns != 'target'].values
y_train = dataset_train.iloc[:, 1].values
X_test = dataset_test.values


# In[ ]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X_train[:,0] = le.fit_transform(X_train[:,0])
X_test[:,0] = le.fit_transform(X_test[:,0])


# In[ ]:


from sklearn.naive_bayes import GaussianNB


# In[ ]:


gnb = GaussianNB()


# In[ ]:


y_pred_gnb = gnb.fit(X_train, y_train).predict(X_test)


# In[ ]:


pd.concat([dataset_test.ID_code, pd.Series(y_pred_gnb).rename('target')], axis = 1).to_csv('gnb_submission.csv')


# In[ ]:


dataset_gnb = pd.concat([dataset_test.ID_code, pd.Series(y_pred_gnb).rename('target')], axis = 1)


# In[ ]:


dataset_gnb.target.value_counts()


# In[ ]:


from sklearn import tree


# In[ ]:


tree = tree.DecisionTreeClassifier()


# In[ ]:


tree.fit(X_train, y_train)


# In[ ]:


y_pred_tree = tree.predict(X_test)


# In[ ]:


dataset_tree = pd.concat((dataset_test.ID_code, pd.Series(y_pred_tree).rename('target')), axis = 1)
dataset_tree


# In[ ]:


dataset_tree.to_csv('tree_submission_final.csv')


# In[ ]:


sample_sub = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/sample_submission.csv')


# In[ ]:


sample_sub


# In[ ]:


#pandas
#numpy
#the train/test split
# Import xgboost
import xgboost as xgb
import pandas as pd


# In[ ]:


xg_cl = xgb.XGBClassifier(objective = 'reg:logistic', n_estimators = 400, seed=132 )


# In[ ]:


xg_cl.fit(X_train, y_train)


# In[ ]:


y_pred_xg = xg_cl.predict(X_test)


# In[ ]:


dataset_xg = pd.concat((dataset_test.ID_code, pd.Series(y_pred_xg).rename('target')), axis = 1)
dataset_xg


# In[ ]:


dataset_xg.to_csv('xgboost_submission.csv')


# In[ ]:



from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold


# In[ ]:


n_estimators = range(50, 400, 50)
param_grid = dict(n_estimators=n_estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
model = xgb.XGBClassifier()


# In[ ]:


grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold)
grid_result = grid_search.fit(X_train, y_train)


# In[ ]:




