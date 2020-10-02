#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


get_ipython().system('pip install --upgrade pip')
get_ipython().system('pip install fastai==0.7.0 ## Based on Fast.ai ML course')


# In[ ]:


get_ipython().system('pip install git+https://github.com/fastai/fastai@2e1ccb58121dc648751e2109fc0fbf6925aa8887')


# In[ ]:


get_ipython().system('apt update && apt install -y libsm6 libxext6')


# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from IPython.display import display
from fastai.imports import *
from fastai.structur import *
from pandas_summary import DataFrameSummary
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
import os
print(os.listdir("../input/"))


# In[ ]:


train = pd.read_csv("../input/santander-customer-transaction-prediction/train.csv")
test = pd.read_csv("../input/santander-customer-transaction-prediction/test.csv")


# In[ ]:


train.describe(include='all')


# In[ ]:


from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler()
df_resampled, y_resampled = ros.fit_sample(train, train['target'])
df_resampled = pd.DataFrame(df_resampled, columns = train.columns)
train['target'].mean(), df_resampled['target'].mean()


# In[ ]:


train_cats(df_resampled)
apply_cats(test, df_resampled)


# In[ ]:


df_trn, y_trn, nas = proc_df(df_resampled, 'target')
df_test, _, _ = proc_df(test, na_dict=nas)


# In[ ]:


X_train, X_valid, y_train, y_valid = train_test_split(df_trn, y_trn, test_size=0.33, random_state=42)


# In[ ]:


from sklearn.metrics import roc_auc_score

def print_score(m):
    res = [roc_auc_score(m.predict(X_train), y_train), roc_auc_score(m.predict(X_valid), y_valid)]
    print(res)


# In[ ]:


set_rf_samples(100000)  ## To train faster, we can train on a smaller subset
m = RandomForestClassifier(n_jobs=-1, n_estimators = 80, max_depth = 10, min_samples_leaf = 10, min_samples_split = 10)
get_ipython().run_line_magic('time', 'm.fit(X_train, y_train)')


# In[ ]:


get_ipython().run_line_magic('time', 'print_score(m)')


# In[ ]:


pred = m.predict(df_test)


# In[ ]:


submission = pd.read_csv('../input/santander-customer-transaction-prediction/sample_submission.csv')
submission['target'] = pred
submission.to_csv('rf_submission_iter3.csv', index=False)

