#!/usr/bin/env python
# coding: utf-8

# # An attempt to impute missing merchant_id
# 
# If you looked at the transactions table you would notice some of the transactions having `merchant_id` missing. This by itself is very confusing (how would Elo not know that?). However, there is a neat trick which can resolve some of the missing information by cross referencing historic/new transactions!

# In[ ]:


import numpy as np
import pandas as pd
pd.set_option('display.float_format', '{:.10f}'.format)

train = pd.read_csv('../input/train.csv')
historical_transactions = pd.read_csv('../input/historical_transactions.csv').fillna('')
new_merchant_transactions = pd.read_csv('../input/new_merchant_transactions.csv').fillna('')

historical_transactions['purchase_amount'] = np.round(historical_transactions['purchase_amount'] / 0.00150265118 + 497.06,2)
new_merchant_transactions['purchase_amount'] = np.round(new_merchant_transactions['purchase_amount'] / 0.00150265118 + 497.06,2)


# The missing merchant_id problem at first sight does not look too significant:

# In[ ]:


missing = historical_transactions[historical_transactions['merchant_id']==''].shape[0]
total = historical_transactions.shape[0]
print(f"Missing: {missing}, total: {total}, missing ratio: {missing/total}")


# However, if we looked at last months the situation is a little bit different - there are about 2% transactions from unknown merchants in the last month, which may be significant if we were able to deal with that somehow!

# In[ ]:


missing = historical_transactions[(historical_transactions['merchant_id']=='')                                  & (historical_transactions['month_lag']==0)].shape[0]
total = historical_transactions[(historical_transactions['month_lag']==0)].shape[0]
print(f"Missing: {missing}, total: {total}, missing ratio: {missing/total}")


# # Approaching the problem by example

# First, let's start off with a very obvious example of `card_id = C_ID_d57e4ddab0`.

# In[ ]:


historical_transactions[historical_transactions['card_id']=='C_ID_d57e4ddab0'].sort_values('purchase_date').tail(10)


# In[ ]:


new_merchant_transactions[new_merchant_transactions['card_id']=='C_ID_d57e4ddab0'].sort_values('purchase_date')


# As you can see, the last "unknown" transaction from `historical_transactions` share the same transaction property (even transaction day of month) as the single transaction from `new_merchant_transactions`. Now this is where you can make the assumption, that these in fact are the transactions from the very same merchant! 
# 
# This assumption can either be correct and wrong (I am not claiming it is right!)
# 
# Given this assumption we can try to reference all the missing merchants in `historical_transactions` with merchants from `new_merchant_transactions`!
# 
# Let's create a dictionary, where we have a unique merchant assigned to the combination of all the fields in the transactions table:

# In[ ]:


fields = ['card_id','city_id','category_1','installments','category_3',          'merchant_category_id','category_2','state_id','subsector_id']
new_merchants = new_merchant_transactions[fields + ['merchant_id']].drop_duplicates()
new_merchants = new_merchants.loc[new_merchants['merchant_id']!='']


# In[ ]:


# take only unique merchants for the `fields` combination
uq_new_merchants = new_merchants.groupby(fields)['merchant_id'].count().reset_index(name = 'n_merchants')
uq_new_merchants = uq_new_merchants.loc[uq_new_merchants['n_merchants']==1]
uq_new_merchants = uq_new_merchants.merge(new_merchants, on = fields)
uq_new_merchants.drop('n_merchants', axis=1, inplace=True)

# rename the merchant_id so we can join it more easily later on
uq_new_merchants.columns = fields + ['imputed_merchant_id']

uq_new_merchants.head()


# At this point we have a unique assignment for `fields` to a `merchant_id`. All we need to do at this point, use this information for `historical_transactions` table.

# In[ ]:


historical_transactions = historical_transactions.merge(uq_new_merchants, on = fields, how = 'left')


# In[ ]:


# make the actual imputation for the merchant_id field
historical_transactions.loc[(historical_transactions['merchant_id']=='') & (~pd.isnull(historical_transactions['imputed_merchant_id'])), 'merchant_id'] = historical_transactions.loc[(historical_transactions['merchant_id']=='') & (~pd.isnull(historical_transactions['imputed_merchant_id'])), 'imputed_merchant_id']


# At this point we are finished - we imputed the missing `merchant_id` fields with some unique `merchant_id` from `new_transactions_table`. Again, it is a speculation that this is the correct approach!
# 
# Let's look how our example `card_id = C_ID_d57e4ddab0` looks now:

# In[ ]:


historical_transactions[historical_transactions['card_id']=='C_ID_d57e4ddab0'].sort_values('purchase_date').tail(10)


# Let's see how the statistics of missing `merchant_id` looks now:

# In[ ]:


missing = historical_transactions[(historical_transactions['merchant_id']=='')                                  & (historical_transactions['month_lag']==0)].shape[0]
total = historical_transactions[(historical_transactions['month_lag']==0)].shape[0]
print(f"Missing: {missing}, total: {total}, missing ratio: {missing/total}")


# So we were able to reduce the missing merchant information of last `month_lag` from 1.9% to 1.6% - that is roughly 15% filled in transactions!

# # tl;dr
# 
# With some assumptions it is possible to impute some of the missing `merchant_id` historic information using the information from the future. 
# 
# This might be useful when calculating features such as number of unique `merchant_id`, average `purhcase_amount` per `merchant_id`, `merchant_id` patterns, etc.
# 
# ## Hope you enjoyed the content - upvote if you found this useful!
