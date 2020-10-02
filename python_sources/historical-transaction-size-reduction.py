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


import pandas as pd


# In[ ]:


hist = pd.read_csv('../input/historical_transactions.csv')


# In[ ]:


hist.info(memory_usage='deep')


# 13.1 GB is huge memory foot print. Let's see if we can smartly use dtypes and reduce size. 29,112,361 total rows and 14 columns.

# In[ ]:


hist.head()


# In[ ]:


hist.authorized_flag.value_counts()


# Convert to category type as it will have less memory need.

# In[ ]:


hist.authorized_flag = hist.authorized_flag.astype('category')
hist.info(memory_usage='deep')


# 1.8GB reduction. On right path.. Let's see how far we can optimise this.

# In[ ]:


hist.category_1.value_counts()


# In[ ]:


hist.category_1 = hist.category_1.astype('category')


# In[ ]:


hist.category_2.value_counts()


# It is tempting to change it to int. However if we see the data dictionary, it is just anonymised category like other categories. So 1.0 does not necessarily mean better/worst than 2.0. So it is similar to category 1. So let's use 'category' dtype.

# In[ ]:


hist.category_2 = hist.category_2.astype('category')


# In[ ]:


hist.category_3.value_counts()


# In[ ]:


hist.category_3 = hist.category_3.astype('category')
hist.info(memory_usage= 'deep')


# In[ ]:


hist.city_id.max(),hist.city_id.min(),sum(hist.city_id.isnull())


# In[ ]:


hist.city_id = hist.city_id.astype('int16')
hist.city_id.max(),hist.city_id.min(),sum(hist.city_id.isnull())


# In[ ]:


hist.installments.max(),hist.installments.min(),sum(hist.installments.isnull())


# In[ ]:


hist.installments = hist.installments.astype('int16')
hist.installments.max(),hist.installments.min(),sum(hist.installments.isnull())


# In[ ]:


hist.month_lag.max(),hist.month_lag.min(),sum(hist.month_lag.isnull())


# In[ ]:


hist.month_lag = hist.month_lag.astype('int16')
hist.month_lag.max(),hist.month_lag.min(),sum(hist.month_lag.isnull())


# In[ ]:


hist.state_id.max(),hist.state_id.min(),sum(hist.state_id.isnull())


# In[ ]:


hist.state_id = hist.state_id.astype('int16')
hist.state_id.max(),hist.state_id.min(),sum(hist.state_id.isnull())


# In[ ]:


hist.subsector_id.max(),hist.subsector_id.min(),sum(hist.subsector_id.isnull())


# In[ ]:


hist.subsector_id = hist.subsector_id.astype('int8')
hist.subsector_id.max(),hist.subsector_id.min(),sum(hist.subsector_id.isnull())


# In[ ]:


hist.info(memory_usage='deep')


# 6.7 GB... Almost 50% reduction. There is no more straight downcast options available.  The remaining option is card_id and merchant_id. They are anonymised string objects. One option to make these workable is replace sequential integers for each uique ids (both merchant and card ids). Slice these mapping of objects and int mappings in other dataframes which can be later used for joining with other datasets.

# In[ ]:


unique_merchants = pd.DataFrame(hist.merchant_id.value_counts())
unique_merchants.reset_index(inplace = True)
unique_merchants.reset_index(inplace = True)
unique_merchants.columns = ['merchant_id_code','merchant_id','count']
unique_merchants.drop(columns = ['count'],inplace = True)
unique_merchants.head()


# In[ ]:


sum(hist.merchant_id.isnull())


# 1. merge unique_merchants with hist df. drop merchant_id from hist df. fill nulls with -1 and make the merchant_id_code as int.

# In[ ]:


hist = hist.merge(unique_merchants, on = 'merchant_id',how = 'left')
hist.drop(columns = ['merchant_id'],inplace = True)
hist.merchant_id_code.fillna(value = -1,inplace = True)
hist.merchant_id_code = hist.merchant_id_code.astype('int32')


# In[ ]:


unique_cards = pd.DataFrame(hist.card_id.value_counts())
unique_cards.reset_index(inplace = True)
unique_cards.reset_index(inplace = True)
unique_cards.columns = ['card_id_code','card_id','count']
unique_cards.drop(columns = ['count'],inplace = True)
unique_cards.head()


# In[ ]:


hist = hist.merge(unique_cards, on = 'card_id',how = 'left')
hist.drop(columns = ['card_id'],inplace = True)
hist.card_id_code.fillna(value=-1,inplace = True)
hist.card_id_code = hist.card_id_code.astype('int32')
hist.info(memory_usage='deep')


# In[ ]:


unique_merchants.info(memory_usage='deep')


# In[ ]:


unique_cards.info(memory_usage='deep')


# 13.1 GB df is converted into 3.3 gb df and 2 dfs with 25mb. So total reduction of 74% reduction !!! That is pretty good progress. Now time to do EDA.....

# In[ ]:




