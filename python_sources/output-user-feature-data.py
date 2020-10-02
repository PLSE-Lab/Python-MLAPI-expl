#!/usr/bin/env python
# coding: utf-8

# In[5]:


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
import gc
import math
import matplotlib.pyplot as plt
import os
import hypertools as hyp
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
os.listdir('../input')
os.listdir('../input/tencent-ad')


# In[2]:


def describe(data_train):
    # print every location count who is null
    print("data is null")
    print(data_train.isnull().sum())
    # print every location count who is not null
    # show column data info
    print(data_train.head(2))
    print("data info")
    print(data_train.info())
    print("data shape")
    print(data_train.shape)
    print(data_train.dtypes)
    print("data type")


# In[3]:


train_user_dir ="../input/norton-tencent-social-ad/user_data"
# user feature data
data_train_user=pd.read_table(train_user_dir,header=None,sep='\t',
                              names=["userId","age","gender","area","status","education","consuptionAbility","device","work","connectionType","behavior"])
print ("ad user")
describe(data_train_user)


# In[7]:


data_train_user.drop(['userId','area','status','work','behavior'],axis=1,inplace=True)
X = StandardScaler().fit_transform(data_train_user[['age']])
data_train_user['age_std'] = X
data_train_user.drop(['age'],axis=1,inplace=True)
data_train_user.head()


# In[8]:


#gridsearch
parm = {'eps':np.arange(0.0001,10,1),'min_samples':np.arange(1,100000,10000)}


# In[10]:


# Compute DBSCAN
db = DBSCAN()
db_model = GridSearchCV(db,parm,cv=3,scoring='v_measure_score',n_jobs=-1)
db_model.fit(data_train_user)
print(db_model.best_params_)
print(db_model.best_score_)


# In[ ]:


db_best = db_model.predict(data_train_user)
core_samples_mask = np.zeros_like(db_best.labels_, dtype=bool)
core_samples_mask[db_best.core_sample_indices_] = True
labels = db_best.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(data_train_user, labels))

