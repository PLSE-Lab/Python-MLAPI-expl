#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV
from scipy.stats import norm
import warnings
import datetime
import time


# In[ ]:


#import data file for kaggle
train = pd.read_csv('../input/game-rating/train.csv')
test = pd.read_csv('../input/game-rating/test.csv')


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


train.shape, test.shape


# In[ ]:


train.info() ,
print('\n'),
test.info()


# In[ ]:


train.select_dtypes(include=object).head()


# In[ ]:


test.select_dtypes(include=object).head()


# In[ ]:


fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (18, 6))

# train data 
sns.heatmap(train.isnull(), yticklabels=False, ax = ax[0], cbar=False, cmap='viridis')
ax[0].set_title('Train data')

# test data
sns.heatmap(test.isnull(), yticklabels=False, ax = ax[1], cbar=False, cmap='viridis')
ax[1].set_title('Test data');


# In[ ]:


train.isnull().sum()


# In[ ]:


test.isnull().sum()


# In[ ]:


x =pd.isnull(train['game_released'])
train[x]


# In[ ]:


x =pd.isnull(test['game_released'])
test[x]


# In[ ]:


c =train['game_released'].replace(np.nan , 0, inplace=True )


# In[ ]:


cc =test['game_released'].replace(np.nan , 0, inplace=True )


# In[ ]:


fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (18, 6))

# train data 
sns.heatmap(train.isnull(), yticklabels=False, ax = ax[0], cbar=False, cmap='viridis')
ax[0].set_title('Train data')

# test data
sns.heatmap(test.isnull(), yticklabels=False, ax = ax[1], cbar=False, cmap='viridis')
ax[1].set_title('Test data');


# In[ ]:




