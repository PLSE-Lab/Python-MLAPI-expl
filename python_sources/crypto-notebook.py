#!/usr/bin/env python
# coding: utf-8

# In[2]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[4]:


data = pd.read_csv('../input/all_currencies.csv')
data.head()


# In[16]:


data.groupby(['Symbol']).count()


# In[18]:


data.loc[data['Symbol'] == 'BTC'].tail()


# In[22]:


sns.jointplot(x='Close',y='Market Cap',data=data.loc[data['Symbol'] == 'BTC'])
data_btc = data.loc[data['Symbol'] == 'BTC']


# In[33]:


data_btc.head()

pd.to_datetime(data['Date']).head()


# In[35]:


sns.tsplot(data_btc['Open'], pd.to_datetime(data_btc['Date']))


# In[ ]:




