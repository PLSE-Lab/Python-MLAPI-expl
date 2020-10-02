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


path = '../input/'
hist = pd.read_csv(path + 'historical_transactions.csv', parse_dates = ['purchase_date'])
new = pd.read_csv(path + 'new_merchant_transactions.csv', parse_dates = ['purchase_date'])
merchant = pd.read_csv(path + 'merchants.csv')


# While revewing Kagglers kernels, one kernel caught my eyes! 
# [https://www.kaggle.com/raddar/merchant-id-imputations](http://)
# 
# Actually he was trying to show that we can fill in NAs in `merchant_id` but I got an idea about feature engineering! Come with me :)
# 

# In[ ]:


hist.loc[hist.card_id == 'C_ID_d57e4ddab0'].sort_values('purchase_date')


# In[ ]:


user_df = hist.loc[hist.card_id == 'C_ID_d57e4ddab0'].sort_values('purchase_date')
user_df.loc[user_df.merchant_id == 'M_ID_9139332ccc']


# This card user ('C_ID_d57e4ddab0') has very unique purchasing tendency.
# When it comes to transaction with merchant_id '	M_ID_9139332ccc', regular monthly puchase is notable  on 14th of every month ! 
# Does it imply regular subscription ???
# Okay! Now my goal is to find regular purchase date of every card user and make it as a feature! 
# Let's go 

# # Regular payment FE 

# In[ ]:


# making `purchase_d` column. Slicing date from `purchase_date` column and shifting its datatype into String
hist['purchase_d'] = [x.strftime('%d') for x in pd.DatetimeIndex(hist.purchase_date).date]     


# In[ ]:


hist.head()


# In[ ]:


# Counting frequency of each value from `purchase_d` column and counting values whose frequency is more than one
# return value implies the frequency of regular monthly purchase 
from collections import Counter
def regular_cnt(series): 
    return sum(filter(lambda x: x > 1, Counter(series.tolist()).values()))


# In[ ]:


regular = hist.groupby(['card_id', 'merchant_id']).agg({'purchase_d': [regular_cnt]})


# In[ ]:


from scipy import stats
def mode(x):
    return stats.mode(x)[1][0]


# In[ ]:


regular_df = regular.groupby(['card_id']).agg(['sum', 'mean', 'min', 'max', 'nunique', 'size', mode])


# In[ ]:


regular_df.head()


# In[ ]:


regular_df.columns = ['hist_regular_sum','hist_regular_mean', 'hist_regular_min', 'hist_regular_max', 'hist_regular_nunique', 'hist_regular_size', 'hist_regular_mode']


# In[ ]:


regular_df.head(10)


# In[ ]:


regular_df.to_csv('regular_FE.csv', index = True)


# In[ ]:




