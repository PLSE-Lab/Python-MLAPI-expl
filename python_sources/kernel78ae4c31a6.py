#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# In[ ]:


from kaggle.competitions import twosigmanews


# In[ ]:


env=twosigmanews.make_env()


# In[ ]:


(market_train_df,news_train_df)=env.get_training_data()


# **Market Data**

# In[ ]:


m_dim=market_train_df.shape


# In[ ]:


print(f'Market dataset has  {m_dim[0]}  sample and {m_dim[1]} feature ')


# In[ ]:


market_train_df.head()


# In[ ]:


market_train_df.dtypes


# Market dataset having 13 numerical feature,1 categorical feature,1 TimeSeries and 1 have object or string

# news data

# In[ ]:


news_train_df.head()


# In[ ]:


n_dim=news_train_df.shape


# In[ ]:


print(f'News dataset has  {n_dim[0]}  sample and {n_dim[1]} feature ')


# In[ ]:


news_train_df.dtypes


# 5-categorical
# 4-datetime
# 23-numerical
# 3-object

# In[ ]:


news_train_df['assetName'].unique()


# In[ ]:


news_train_df['assetName'].value_counts()


# In[ ]:


market_train_df['assetName'].unique()


# In[ ]:


market_train_df['assetName'].value_counts()


# 

# In[ ]:


market_train_df.dtypes


# Check %age of nan value

# In[ ]:


market_train_df.isnull().sum()*100/market_train_df.shape[0]


# returnsClosePrevMktres1     0.392344%
# 
# returnsOpenPrevMktres1      0.392540%
# 
# returnsClosePrevMktres10    2.283599%
# 
# returnsOpenPrevMktres10     2.284680%    
# 

# 1st fill the nan value..

# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


plt.figure(figsize=(15,10))
market_train_df['returnsClosePrevMktres1'].plot(kind='box')


# few value is outlier...

# In[ ]:


( market_train_df['returnsClosePrevMktres1']<market_train_df['returnsClosePrevMktres1'].mean()).value_counts()


# In[ ]:


( market_train_df['returnsClosePrevMktres1']<market_train_df['returnsClosePrevMktres1'].median()).value_counts()


# In[ ]:


market_train_df['returnsClosePrevMktres1'].fillna(market_train_df['returnsClosePrevMktres1'].mean(),inplace=True)


# In[ ]:


market_train_df['returnsOpenPrevMktres1'].plot.box()


# Many outlier so fill with median

# In[ ]:


market_train_df['returnsOpenPrevMktres1'].fillna(market_train_df['returnsOpenPrevMktres1'].median(),inplace=True)


# In[ ]:


plt.figure(figsize=(15,10))
market_train_df['returnsClosePrevMktres10'].plot.box()


# Many outlier so fill with median

# In[ ]:


market_train_df['returnsClosePrevMktres10'].fillna(market_train_df['returnsClosePrevMktres10'].median(),inplace=True)


# In[ ]:


plt.figure(figsize=(15,10))
market_train_df['returnsOpenPrevMktres1'].plot.box()


# Outliers having big value so we fill with median

# In[ ]:


market_train_df['returnsOpenPrevMktres10'].fillna(market_train_df['returnsOpenPrevMktres10'].median(),inplace=True)


# In[ ]:


market_train_df.isnull().sum()


# In[ ]:


plt.figure(figsize=(14,7))
plt.plot(market_train_df['time'],market_train_df['volume'],label='Time vs Volume')
plt.xlabel('Time')
plt.ylabel('Volume')
plt.legend()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




