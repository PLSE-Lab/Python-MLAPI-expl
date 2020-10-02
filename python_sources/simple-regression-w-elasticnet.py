#!/usr/bin/env python
# coding: utf-8

# # Ion Switching
# - Identify the number of channels open at each time point
# 
# In this competition, you will be predicting the number of `open_channels` present, based on electrophysiological signal data.
# 
# IMPORTANT: While the time series appears continuous, the data is from discrete batches of 50 seconds long 10 kHz samples (500,000 rows per batch). In other words, the data from 0.0001 - 50.0000 is a different batch than 50.0001 - 100.0000, and thus discontinuous between 50.0000 and 50.0001.
# 
# You can find detailed information about the data from the paper Deep-Channel uses deep neural networks to detect single-molecule events from patch-clamp data.
# 
# ## Contents
# 1. [Introduction](#1.-Introduction)
# 2. [Plot](#2.-Plot)
# 3. [Fit Model](#3.-Fit-Model)
# 4. [Submission](#4.-Submission)

# # 1. Introduction

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection
from sklearn.linear_model import ElasticNet

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


df_train = pd.read_csv('/kaggle/input/liverpool-ion-switching/train.csv')
df_test = pd.read_csv('/kaggle/input/liverpool-ion-switching/test.csv')
df_submission = pd.read_csv('/kaggle/input/liverpool-ion-switching/sample_submission.csv')


# In[ ]:


X_train = df_train[['signal']].values
y_train = df_train['open_channels'].values
X_test = df_test[['signal']].values

df_train.head()


# # 2. Plot

# In[ ]:


plt.figure(figsize=(12, 6))
sns.countplot(x='open_channels', data=df_train)


# In[ ]:


plt.figure(figsize=(12, 6))
sns.kdeplot(df_train['signal'])


# In[ ]:


plt.figure(figsize=(12, 6))
sns.distplot(df_train['signal'])


# In[ ]:


plt.figure(figsize=(12, 12))
sns.jointplot(x='time', y='signal', data=df_train, kind='hex', gridsize=20)


# # 3. Fit Model

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nmodel = ElasticNet(alpha=0.9, l1_ratio=0.1)\nmodel.fit(X_train, y_train)')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'preds = model.predict(X_test)\n\n# preds_reshaped = np.reshape(preds, (int(len(preds)/10), -1))\n# preds_reduced_mean = np.mean(preds_reshaped, axis=1)\n#preds_around = np.around(preds, decimals=1).astype(int)\npreds_around = np.rint(preds).astype(int)')


# In[ ]:


df_submission['open_channels'] = preds_around
df_submission[df_submission['open_channels']<0]['open_channels'] = 0
df_submission.head()


# In[ ]:


sns.countplot(x='open_channels', data=df_train)


# In[ ]:


sns.countplot(x='open_channels', data=df_submission)


# # 4. Submission
# 
# If you use just to_csv() function you will meet an error with 'time'
# because it will lose last 0
# 
# example 500.0010 -> 500.001
# 
# ```
# ERROR: Unable to find 200000 required key values in the 'time' column
# ERROR: Unable to find the required key value '500.0010' in the 'time' column
# ```

# In[ ]:


df_submission.head(10)


# In[ ]:


df_submission['time'] = [ "{:.4f}".format(df_submission['time'].values[x]) for x in range(2000000)]
df_submission.head(10)


# In[ ]:


df_submission.to_csv("submission.csv", index=False)


# In[ ]:




