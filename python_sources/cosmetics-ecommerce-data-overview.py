#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Load everything

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # visualizations
import os


# # Data overview

# ## Read data
# 
# And prepare some filtered datasets. Keep in mind: this file contains about 42M rows, so it loads slowly.

# In[ ]:


# Read data
data_csv = "/kaggle/input/ecommerce-events-history-in-cosmetics-shop/2019-Oct.csv"
raw_data = pd.read_csv(data_csv)

# Get only purchases
only_purchases = raw_data.loc[raw_data.event_type == 'purchase']


# In[ ]:


raw_data.head()


# ## Brands popularity
# 
# Shows the most popular brands (by total sales):

# In[ ]:


# With brands only
purchases_with_brands = only_purchases.loc[only_purchases.brand.notnull()]
top_sellers = purchases_with_brands.groupby('brand').brand.agg([len]).sort_values(by='len', ascending=False)
top_sellers.head(20)


# ## User's journey
# 
# This example shows a common flow of a random user: what she/he looked and what bought.

# In[ ]:


raw_data.loc[raw_data.user_session == "26dd6e6e-4dac-4778-8d2c-92e149dab885"].sort_values(by='event_time')


# ## All events of the same user

# In[ ]:


raw_data.loc[raw_data.user_id == 10280338].sort_values(by='event_time')

