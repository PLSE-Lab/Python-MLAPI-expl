#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Load everything

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # visualizations
import os


# # Data overview

# ### Read data 
# 
# And prepare some filtered datasets. Keep in mind: this file contains about 42M rows, so it loads slowly.

# In[ ]:


# Read data
data_csv = "/kaggle/input/ecommerce-behavior-data-from-multi-category-store/2019-Oct.csv"
raw_data = pd.read_csv(data_csv)

# Get only purchases
only_purchases = raw_data.loc[raw_data.event_type == 'purchase']


# In[ ]:


raw_data.head()


# ### Brands popularity
# 
# Shows the most popular brands (by total sales):

# In[ ]:


# With brands only
purchases_with_brands = only_purchases.loc[only_purchases.brand.notnull()]
top_sellers = purchases_with_brands.groupby('brand').brand.agg([len]).sort_values(by='len', ascending=False)
top_sellers.head(20)


# # User's journey
# 
# This example shows a common flow of a random user: what she/he looked and what bought.
# 
# 
# You can read it like:
# 1. The user checked out several iPhones
# 2. Purchased one iPhone in 1 click (without `cart` event)
# 3. Viewed 2 unknown products of brand `arena`
# 4. Visited some Apple's headphones and purchased one
# 5. After that visited more expensive one but decided to not buy it

# In[ ]:


raw_data.loc[raw_data.user_session == "3c80f0d6-e9ec-4181-8c5c-837a30be2d68"].sort_values(by='event_time')

