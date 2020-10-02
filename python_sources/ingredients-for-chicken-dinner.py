#!/usr/bin/env python
# coding: utf-8

# # How top 1% play the game differently ?

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import seaborn as sns 
import warnings
warnings.filterwarnings("ignore")

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv('../input/train_V2.csv')
df.head()


# In[ ]:


df = df.dropna() #using any startegy


# In[ ]:


# Memory saving function credit to https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    #start_mem = df.memory_usage().sum() / 1024**2
    #print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        return df


# In[ ]:


df = reduce_mem_usage(df)


# In[ ]:


df_solo = df.query('matchType == "solo"')


# In[ ]:


df_solo.drop(['Id', 'groupId', 'matchId'],inplace=True,axis=1)


# In[ ]:


df_solo_winner = df.query('winPlacePerc >= 0.99')
df_solo_noobs  = df.query('teamKills <0.99')


# In[ ]:


df_solo.columns


# In[ ]:


value = 'rideDistance'
sns.distplot(df_solo_noobs[value])
sns.distplot(df_solo_winner[value])
plt.xlim(0,10000)
print(df_solo_noobs[value].median())
print(df_solo_winner[value].median())


# In[ ]:


value = 'weaponsAcquired'
sns.distplot(df_solo_noobs[value])
sns.distplot(df_solo_winner[value])
plt.xlim(0,15)
print(df_solo_noobs[value].median())
print(df_solo_winner[value].median())


# In[ ]:


value = 'headshotKills'
sns.distplot(df_solo_noobs[value])
sns.distplot(df_solo_winner[value])
plt.xlim(0,6)
print(df_solo_noobs[value].median())
print(df_solo_winner[value].median())


# In[ ]:


value = 'walkDistance'
sns.distplot(df_solo_noobs[value])
sns.distplot(df_solo_winner[value])
plt.xlim(0,6000)
print(df_solo_noobs[value].median())
print(df_solo_winner[value].median())


# In[ ]:


value = 'kills'
sns.distplot(df_solo_noobs[value])
sns.distplot(df_solo_winner[value])
plt.xlim(0,15)
print(df_solo_noobs[value].median())
print(df_solo_winner[value].median())


# In[ ]:


value = 'heals'
sns.distplot(df_solo_noobs[value])
sns.distplot(df_solo_winner[value])
plt.xlim(0,10)
print(df_solo_noobs[value].median())
print(df_solo_winner[value].median())


# In[ ]:


value = 'damageDealt'
sns.distplot(df_solo_noobs[value])
sns.distplot(df_solo_winner[value])
plt.xlim(0,1300)
print(df_solo_noobs[value].median())
print(df_solo_winner[value].median())


# In[ ]:


value = 'boosts'
sns.distplot(df_solo_noobs[value])
sns.distplot(df_solo_winner[value])
plt.xlim(0,10)
print(df_solo_noobs[value].median())
print(df_solo_winner[value].median())


# In[ ]:


value = 'assists'
sns.distplot(df_solo_noobs[value])
sns.distplot(df_solo_winner[value])
plt.xlim(0,5)
print(df_solo_noobs[value].median())
print(df_solo_winner[value].median())


# 
# 
# Please upvote this kernel so that it reaches the top of the chart and is easily locatable by new users. Your comments on how we can improve this kernel is welcome. Thanks.
# 
