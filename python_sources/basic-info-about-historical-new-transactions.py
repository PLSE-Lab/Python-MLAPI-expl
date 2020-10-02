#!/usr/bin/env python
# coding: utf-8

# **Data Description** has little information about `historical_transactions.csv` and `new_merchant_transactions.csv`.
# >  - historical_transactions.csv - up to 3 months' worth of historical transactions for each card_id
# >  - new_merchant_transactions.csv - two months' worth of data for each card_id containing ALL purchases that card_id made at merchant_ids that were not visited in the historical data.
# 
# This kernel do same statistic about this two files.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


hist_trans = pd.read_csv('../input/historical_transactions.csv')
new_trans = pd.read_csv('../input/new_merchant_transactions.csv')


# In[ ]:


hist_trans.shape, new_trans.shape


# ### 1. Overall time range

# In[ ]:


hist_trans.purchase_date = pd.to_datetime(hist_trans.purchase_date)
new_trans.purchase_date = pd.to_datetime(new_trans.purchase_date)


# In[ ]:


hist_trans.purchase_date.agg([np.min, np.max])


# In[ ]:


new_trans.purchase_date.agg([np.min, np.max])


# So, we know that `historical_transactions.csv` records transactions from 2017/01/01 to 2018/02/28, and `new_merchant_transactions.csv` records transactions from 2017/03/01 to 2018/04/30.

# ### 2. Time range for each `card_id`

# In[ ]:


hist_time_range = hist_trans.groupby('card_id', sort=False)['purchase_date'].agg([np.min, np.max])
new_time_range = new_trans.groupby('card_id', sort=False)['purchase_date'].agg([np.min, np.max])


# In[ ]:


hist_time_range.head()


# In[ ]:


new_time_range.head()


# In[ ]:


hist_time_range.shape, new_time_range.shape


# In[ ]:


set(hist_time_range.index.tolist()) - set(new_time_range.index.tolist())


# In[ ]:


set(new_time_range.index.tolist()) - set(hist_time_range.index.tolist())


# `card_id` in `new_merchant_transactions.csv` also in `historical_transactions.csv`,  but the reverse is not the case.

# In[ ]:


time_range = hist_time_range.join(new_time_range, how='right', lsuffix='_hist', rsuffix='_new')


# In[ ]:


time_range.head()


# In[ ]:


time_range.shape


# Does historical_transactions early than new_transactions for all `card_id`?

# In[ ]:


all(time_range.amax_hist < time_range.amin_new)


# How long is `historical_transactions/new_merchant_transactions` for each `card_id`?

# In[ ]:


hist_month_num = (time_range.amax_hist - time_range.amin_hist).apply(lambda x: x.days / 30)
new_month_num = (time_range.amax_new - time_range.amin_new).apply(lambda x: x.days / 30)


# In[ ]:


plt.hist(hist_month_num.values, bins=50)
plt.title('Historical transactions time range')
plt.xlabel('Month')
plt.ylabel('Counts')


# In[ ]:


plt.hist(new_month_num.values, bins=50)
plt.title('New merchant transactions time range')
plt.xlabel('Month')
plt.ylabel('Counts')


# Well, historical transactions lasts more than 3 months, which is not consistent with **Data Description**.(am I wrong? if someone know the reason, please tell me.). new merchant transactions did last up to 2 months.

# ### 3. New merchant

# `new_merchant_transactions`: ALL purchases that card_id made at `merchant_id`s that were not visited in the historical data. Let's check it.

# We'd better remove NA merchant_id

# In[ ]:


hist_trans.dropna(axis=0, how='any', inplace=True)
new_trans.dropna(axis=0, how='any', inplace=True)


# In[ ]:


hist_trans.shape, new_trans.shape


# In[ ]:


hist_merchant_set = hist_trans.groupby('card_id')['merchant_id'].apply(lambda x: set(x.tolist()))
new_merchant_set = new_trans.groupby('card_id')['merchant_id'].apply(lambda x: set(x.tolist()))


# In[ ]:


hist_merchant_set.head()


# In[ ]:


merchant_set = pd.concat([hist_merchant_set, new_merchant_set], axis=1, join='inner')


# In[ ]:


merchant_set.columns = ['hist', 'new']


# In[ ]:


merchant_set.head()


# In[ ]:


intersection = merchant_set.apply(lambda x: len(set.intersection(x['hist'], x['new'])), axis=1)


# In[ ]:


intersection.head()


# In[ ]:


any(intersection.values)


# There is no intersections between `historical_transactions.merchant_id` and `new_merchant_transactions.merchant_id` for every `card_id`.

# Hope this will help, thanks!

# In[ ]:




