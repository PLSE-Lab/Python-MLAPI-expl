#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas_profiling import ProfileReport

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv("/kaggle/input/cattells-16-personality-factors/16PF/data.csv", sep="\t")


# In[ ]:


data.head()


# In[ ]:


data.shape


# In[ ]:


data[data['country'] == 'US'].shape


# In[ ]:


data[data['gender'] == 1].shape


# In[ ]:


#%%time
#profile = ProfileReport(data, title='Pandas Profiling Report', html={'style':{'full_width':True}})


# In[ ]:


#profile


# In[ ]:


np.unique(data['gender'].values)


# In[ ]:


gendered_data = data[(data['gender'] == 1) | (data['gender'] == 2)]


# In[ ]:


gendered_data.shape


# In[ ]:


gendered_data['gender'] = gendered_data['gender'].values -1


# In[ ]:


gendered_data['gender'].head(20)


# In[ ]:


gendered_data.columns


# In[ ]:


gendered_data.columns[:-6]


# In[ ]:


features = gendered_data.columns[:-6]


# In[ ]:


gendered_data[features].values.max()


# In[ ]:


gendered_data[features].values.min()


# In[ ]:


gendered_data[features] = gendered_data[features].values/5.


# In[ ]:


gendered_data[features].head()


# In[ ]:


gendered_data[features].std(axis=1)


# In[ ]:


gendered_data['std'] = gendered_data[features].std(axis=1)


# In[ ]:


gendered_data = gendered_data[gendered_data['std'] > 0.0]


# In[ ]:


gendered_data.shape


# In[ ]:


X = gendered_data[features].values
Y = gendered_data['gender'].values


# In[ ]:


np.mean(Y)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)


# In[ ]:


lr = LogisticRegression(C=20)
lr.fit(X_train, y_train)
preds_0 = lr.predict(X_test)
preds_1 = lr.predict_proba(X_test)[:,1]


# In[ ]:


accuracy_score(y_test, preds_0)


# In[ ]:


0.7967605488496854


# In[ ]:


roc_auc_score(y_test, preds_1)


# In[ ]:


0.870735955752324


# In[ ]:




