#!/usr/bin/env python
# coding: utf-8

# 1. Scale the features to uniform range.
# 2. Sort all the features within a sample from min to max.
# 3. Plot the difference of min and max of new sorted features.
# 
# Plotting code copied from Youhanlee's [kernel](https://www.kaggle.com/youhanlee/yh-eda-i-want-to-see-all)

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import KFold
import warnings
import gc
import time
import sys
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')
from sklearn import metrics

plt.style.use('seaborn')
sns.set(font_scale=2)
pd.set_option('display.max_columns', 500)


# In[ ]:


get_ipython().run_cell_magic('time', '', "raw_train = pd.read_csv('../input/train.csv')\ntrain = raw_train")


# In[ ]:


target_mask = train['target'] == 1
non_target_mask = train['target'] == 0 


# In[ ]:


from scipy.stats import ks_2samp


# In[ ]:


train = raw_train.copy(deep=True)
cols = [f for f in train.columns if f not in ['ID_code', 'target']]
sc = MinMaxScaler(feature_range=(1, 2))
scaled_train = sc.fit_transform(train[cols])
print(scaled_train.shape)
for i in range(200):
    train[f'var_{i}'] = scaled_train[:,i] 

train[cols] = np.sort(train[cols])


train.describe()


# In[ ]:


for i in range(200):
    f = f'var_{i}'
    fld = f'bin_{i}'
    train[fld] = np.around(((train[f] - 1.0)*200))/200 + 1.0


# In[ ]:


def plot_col(col):
    statistic, pvalue = ks_2samp(train.loc[non_target_mask, col], train.loc[target_mask, col])
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    sns.kdeplot(train.loc[non_target_mask, col], ax=ax, label='Target == 0')
    sns.kdeplot(train.loc[target_mask, col], ax=ax, label='Target == 1')

    ax.set_title('name: {}, statistics: {:.5f}, pvalue: {:5f}'.format(col, statistic, pvalue))
    plt.show()


# In[ ]:


def plot_diff(col1, col2, show_each=False):
    n = f'diff_{col1}-{col2}'
    print(n)
    if(show_each):
        plot_col(col1)
        plot_col(col2)
    
    train[n] = train[col1] - train[col2]
        
    plot_col(n)


# In[ ]:


plot_col('bin_15')


# In[ ]:


plot_diff('var_184', 'var_15')


# In[ ]:


for i in range(10):
    plot_diff(f'var_{i}', f'var_{199-i}', True)


# In[ ]:


for i in range(10):
    plot_diff(f'bin_{i}', f'bin_{199-i}', False)

