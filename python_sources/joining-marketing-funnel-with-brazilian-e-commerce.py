#!/usr/bin/env python
# coding: utf-8

# # How to join Marketing Funnel Dataset with Brazilian E-Commerce Public Dataset
# Olist has published the Brazilian E-Commerce Public Dataset a few months ago, and now we are publishing this Marketing Funnel Dataset. You are able to join both datasets and observe the customer journey, since the moment he first entered our marketing funnel.
# 
# ## Files from both datasets

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

print('### Marketing Funnel by Olist ###')
for idx, file in enumerate(os.listdir('../input/marketing-funnel-olist')):
    print(idx, '-', file)
print('\n---------------------------------------------\n')

print('### Brazilian E-Commerce Public Dataset by Olist ###')
for idx, file in enumerate(os.listdir('../input/brazilian-ecommerce')):
    print(idx, '-', file)


# ## Data Schema
# 
# This dataset may be easily linked to the Brazilian Ecommerce Public Dataset, just follow the data schema presented bellow:
# ![](https://i.imgur.com/Jory0O3.png)

# In[ ]:


# leads dataset
mql = pd.read_csv('../input/marketing-funnel-olist/olist_marketing_qualified_leads_dataset.csv')
mql.head(10)


# In[ ]:


# closed deals dataset
cd = pd.read_csv('../input/marketing-funnel-olist/olist_closed_deals_dataset.csv')
cd.head(10)


# In[ ]:


# marketing funnel dataset (NaNs are leads that did not close a deal)
mf = mql.merge(cd, on='mql_id', how='left')
mf.head(10)


# In[ ]:


# sellers dataset
sellers = pd.read_csv('../input/brazilian-ecommerce/olist_sellers_dataset.csv')
sellers.head(10)


# In[ ]:


# marketing funnel merged with sellers (this way you get seller location)
mf_sellers = mf.merge(sellers, on='seller_id', how='left')
mf_sellers.head(10)


# In[ ]:


# order items dataset
items = pd.read_csv('../input/brazilian-ecommerce/olist_order_items_dataset.csv')
items.head(10)


# In[ ]:


# marketing funnel merged with items (this way you get products sold by sellers)
mf_items = mf.merge(items, on='seller_id', how='left')
mf_items.head(10)


# We hope you enjoy this new dataset and its interaction with the Brazilian E-commerce Public Dataset!

# In[ ]:




