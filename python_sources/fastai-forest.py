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


get_ipython().system('pip install fastai==0.7')


# In[ ]:


from fastai.imports import *
from fastai.structured import *
from pandas_summary import DataFrameSummary
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from IPython.display import display


# In[ ]:


train_data = pd.read_csv('/kaggle/input/learn-together/train.csv',index_col='Id')
test_data = pd.read_csv('/kaggle/input/learn-together/test.csv',index_col='Id')


# In[ ]:


# num_train = train_data.shape[0]


# In[ ]:


# target = train_data['Cover_Type']
# train_data = train_data.drop('Cover_Type',axis=1)
# all_data = train_data.append(test_data)


# In[ ]:


# from sklearn.mixture import GaussianMixture
# GM = GaussianMixture(n_components=8)
# GM.fit(all_data)


# In[ ]:


# plt.hist(GM)


# In[ ]:


train_cats(train_data)


# In[ ]:


train , y, nas = proc_df(train_data,'Cover_Type')


# In[ ]:


apply_cats(test_data, train_data)


# In[ ]:


test, _ ,nas_ = proc_df(test_data)


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


# rf = RandomForestClassifier(n_jobs=-1)
# from sklearn.model_selection import cross_val_score


# In[ ]:


X_train,X_val,y_train,y_val = train_test_split(train,y,test_size=0.3,random_state=10)


# In[ ]:


# rf.fit(X_train,y_train)
# metrics.accuracy_score(rf.predict(X_val),y_val)


# In[ ]:


# rf = RandomForestClassifier(n_estimators= 20, n_jobs=-1)
# rf.fit(X_train, y_train)
# metrics.accuracy_score(rf.predict(X_val),y_val)


# In[ ]:


# rf = RandomForestClassifier(n_estimators=40, n_jobs=-1)
# rf.fit(X_train, y_train)
# metrics.accuracy_score(rf.predict(X_val),y_val)


# In[ ]:


# rf = RandomForestClassifier(n_estimators=60, n_jobs=-1)
# rf.fit(X_train,y_train)
# print(metrics.accuracy_score(rf.predict(X_train),y_train))
# print(metrics.accuracy_score(rf.predict(X_val),y_val))


# In[ ]:


# rf = RandomForestClassifier(n_estimators=40,min_samples_leaf=5,n_jobs=-1)
# rf.fit(X_train,y_train)
# print(metrics.accuracy_score(rf.predict(X_train),y_train))
# print(metrics.accuracy_score(rf.predict(X_val),y_val))


# In[ ]:


rf = RandomForestClassifier(n_estimators=40,min_samples_leaf=5,max_features=0.5,n_jobs=-1)
rf.fit(X_train,y_train)
print(metrics.accuracy_score(rf.predict(X_train),y_train))
print(metrics.accuracy_score(rf.predict(X_val),y_val))


# In[ ]:


preds = rf.predict(test)


# In[ ]:


submit = pd.DataFrame({'Cover_Type':preds},index=test.index)


# In[ ]:


submit.to_csv('submission.csv')


# In[ ]:




