#!/usr/bin/env python
# coding: utf-8

# #### This kernel shows how to estimate standard deviations associated with gaussian distribution of each open channels using the regions of high confidence interval (Mean-0.1) to (Mean+0.1). 
# ### This feature works great for HMM & Random Forest (My 133rd place solution).

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import lightgbm as lgb
from sklearn.metrics import classification_report

import seaborn as sns
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


from IPython.display import display, HTML
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')
from sklearn.metrics import f1_score
train_df = pd.read_csv("/kaggle/input/scaling-3/new_train.csv")
test_df = pd.read_csv("/kaggle/input/scaling-3/new_test.csv")
train_df['batch']=((train_df.time-0.0001)//50).astype(int)
test_df['batch']=((test_df.time-0.0001)//50).astype(int)
train_df['mini_batch']=((train_df.time-0.0001)//10).astype(int)
test_df['mini_batch']=((test_df.time-0.0001)//10).astype(int)
train_df['mini_mini_batch']=((train_df.time-0.0001)//0.5).astype(int)
test_df['mini_mini_batch']=((test_df.time-0.0001)//0.5).astype(int)


# In[ ]:


train_means = train_df.groupby('open_channels').signal.agg('mean').reset_index()
train_means['lb'] = train_means.signal-0.1  #Lower Bound
train_means['ub'] = train_means.signal+0.1  #Upper Bound
train_means


# In[ ]:


train_df['high_ci'] = 0
for oc in range(11):
    train_df.loc[(train_df.signal>train_means.loc[oc,'lb']) & (train_df.signal<train_means.loc[oc,'ub']),'high_ci'] += 1
train_df.high_ci.value_counts()


# In[ ]:


cis = train_df.groupby('mini_batch').high_ci.mean().reset_index()
cis


# In[ ]:


stds = train_df.groupby(['mini_batch','open_channels']).signal.agg(['std','count']).reset_index()
stds = stds.loc[stds.groupby(['mini_batch'], sort=False)[('count')].idxmax().values].drop('open_channels',axis=1)
stds


# In[ ]:


stds = pd.merge(cis,stds,on='mini_batch')
stds


# In[ ]:


stds.plot.scatter('std','high_ci')


# In[ ]:


test_df['high_ci'] = 0
for oc in range(11):
    test_df.loc[(test_df.signal>train_means.loc[oc,'lb']) & (test_df.signal<train_means.loc[oc,'ub']),'high_ci'] += 1
test_df.high_ci.value_counts()


# In[ ]:


test_std = test_df.groupby('mini_batch').high_ci.mean().reset_index()
test_std


# In[ ]:


stds[['high_ci2']] = stds[['high_ci']]**2


# In[ ]:


from sklearn.linear_model import LinearRegression,Ridge
model = LinearRegression().fit(stds[['high_ci','high_ci2']],stds[['std']])
model.score(stds[['high_ci','high_ci2']],stds[['std']])


# In[ ]:


test_std[['high_ci2']] = test_std[['high_ci']]**2
test_std['pred'] = model.predict(test_std[['high_ci','high_ci2']])
test_std


# In[ ]:


stds['pred'] = model.predict(stds[['high_ci','high_ci2']])
stds


# In[ ]:


stds.set_index('mini_batch').pred.to_dict()


# In[ ]:


test_std.set_index('mini_batch').pred.to_dict()


# In[ ]:


(stds['std'] - stds['pred']).abs().sort_values(ascending=False)


# In[ ]:




