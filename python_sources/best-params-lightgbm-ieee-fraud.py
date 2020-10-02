#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

import os
for dirname, _, filenames in os.walk('/kaggle/input/pickle-ieee'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


train = pd.read_pickle('/kaggle/input/pickle-ieee/Train.pkl')
test = pd.read_pickle('/kaggle/input/pickle-ieee/Test.pkl')


# In[ ]:


y = train['isFraud']
del train['isFraud']


# # Model

# **Finding best parameters using RandomizedSearchCV**

# In[ ]:


import lightgbm as lgbm


# In[ ]:


from sklearn.model_selection import RandomizedSearchCV


# In[ ]:


params = {
    'num_leaves': [500,600,700,800],
    'feature_fraction': list(np.arange(0.1,0.5,0.1)),
    'bagging_fraction': list(np.arange(0.1,0.5,0.1)),
    'min_data_in_leaf': [100,120,140,160],
    'learning_rate': [0.05],
    'reg_alpha': list(np.arange(0.1,0.5,0.1)),
    'reg_lambda': list(np.arange(0.1,0.5,0.1)),
}


# In[ ]:


model = lgbm.LGBMClassifier(random_state=42,metric='auc',verbosity=-1,objective='binary',max_depth=-1)


# In[ ]:


grid = RandomizedSearchCV(model,param_distributions=params,n_iter=15,cv=3,scoring='roc_auc')


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(
    train, y, test_size=0.15, random_state=42)


# In[ ]:


grid.fit(X_train,y_train)


# In[ ]:


grid.best_params_


# In[ ]:


grid.best_score_


# In[ ]:


from sklearn.metrics import classification_report, roc_auc_score


# In[ ]:


print(classification_report(y_test,grid.predict(X_test)))


# In[ ]:


print(roc_auc_score(y_test,grid.predict_proba(X_test)[:,1]))

