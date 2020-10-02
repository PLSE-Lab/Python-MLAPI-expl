#!/usr/bin/env python
# coding: utf-8

# Thanks to https://www.kaggle.com/tunguz for sharing this!
# Metrics matter
# Compare accuracy, Cohen's kappa and ....QUADRATIC Cohen's kappa....

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


from sklearn.linear_model import Ridge


# In[ ]:


train = pd.read_csv('/kaggle/input/liverpool-ion-switching/train.csv')
submission = pd.read_csv('/kaggle/input/liverpool-ion-switching/sample_submission.csv')


# In[ ]:


train.head()


# In[ ]:


train.tail()


# In[ ]:


train.loc[49000:50010,:]


# In[ ]:


train.shape


# In[ ]:


train['open_channels'].min()


# In[ ]:


train_time = train['time'].values


# In[ ]:


train_time_0 = train_time[:50000]


# In[ ]:


for i in range(1,100):
    train_time_0 = np.hstack([train_time_0, train_time[i*50000:(i+1)*50000]])


# In[ ]:


train['time'] = train_time_0


# In[ ]:


test = pd.read_csv('/kaggle/input/liverpool-ion-switching/test.csv')


# In[ ]:


test.head()


# In[ ]:


test.tail()


# In[ ]:


test.shape


# In[ ]:


train_time_0 = train_time[:50000]
for i in range(1,40):
    train_time_0 = np.hstack([train_time_0, train_time[i*50000:(i+1)*50000]])
test['time'] = train_time_0


# The following signal processing parts are taken from the following Khoi Nguyen kernel: https://www.kaggle.com/suicaokhoailang/an-embarrassingly-simple-baseline-0-960-lb

# In[ ]:


n_groups = 100
train["group"] = 0
for i in range(n_groups):
    ids = np.arange(i*50000, (i+1)*50000)
    train.loc[ids,"group"] = i


# In[ ]:


n_groups = 40
test["group"] = 0
for i in range(n_groups):
    ids = np.arange(i*50000, (i+1)*50000)
    test.loc[ids,"group"] = i


# In[ ]:


train['signal_2'] = 0
test['signal_2'] = 0


# In[ ]:


n_groups = 100
for i in range(n_groups):
    sub = train[train.group == i]
    signals = sub.signal.values
    imax, imin = math.floor(np.max(signals)), math.ceil(np.min(signals))
    signals = (signals - np.min(signals))/(np.max(signals) - np.min(signals))
    signals = signals*(imax-imin)
    train.loc[sub.index,"signal_2"] = [0,] +list(np.array(signals[:-1]))


# In[ ]:


n_groups = 40
for i in range(n_groups):
    sub = test[test.group == i]
    signals = sub.signal.values
    imax, imin = math.floor(np.max(signals)), math.ceil(np.min(signals))
    signals = (signals - np.min(signals))/(np.max(signals) - np.min(signals))
    signals = signals*(imax-imin)
    test.loc[sub.index,"signal_2"] = [0,] +list(np.array(signals[:-1]))


# In[ ]:


signals.shape


# In[ ]:


#X = train[['time', 'signal_2']].values
X = train[['signal_2']].values
y = train['open_channels'].values


# In[ ]:


model = Ridge()
model.fit(X, y)


# In[ ]:


train_preds = model.predict(X)
train_preds = np.clip(train_preds, 0, 10)
train_preds = train_preds.astype(int)


# In[ ]:


from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import accuracy_score
gt_labels=train.values[:,2]
print(gt_labels[:5])
print(train_preds[:5])
print(accuracy_score(gt_labels,train_preds))
print(cohen_kappa_score(gt_labels,train_preds))
print(cohen_kappa_score(gt_labels,train_preds,weights="quadratic"))


# In[ ]:




