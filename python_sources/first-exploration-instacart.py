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

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


aisles = pd.read_csv('../input/aisles.csv')
departments = pd.read_csv('../input/departments.csv')
prior = pd.read_csv('../input/order_products__prior.csv')
train = pd.read_csv('../input/order_products__train.csv')
orders = pd.read_csv('../input/orders.csv')
products = pd.read_csv('../input/products.csv')
sample = pd.read_csv('../input/sample_submission.csv')


# Look at aisles and departments first
# ------------------------------------

# In[ ]:


print(aisles[:10])
print("*"*50)
print(len(aisles))
print(len(aisles['aisle'].unique()))
# 134 rows, each a unique aisle


# In[ ]:


print(departments)
print("*"*50)
print(len(departments))
print(len(departments['department'].unique()))
# 21 rows, each a unique department


# Now look at prior data
# ----------------------

# In[ ]:


prior[:15]


# In[ ]:


prior.shape


# **order_id**

# In[ ]:


print(len(prior['order_id'].unique())) # 3,214,874 unique order_ids
print(prior['order_id'].min(),prior['order_id'].max()) # range from 2 to 3421083
# top 20 order_ids by number of rows:
print(prior.groupby('order_id')['product_id'].count().reset_index().sort_values('product_id',ascending=False)[:20])


# **product_id**

# In[ ]:


print(len(prior['product_id'].unique())) # 49,677 unique product_ids
print(prior['product_id'].min(),prior['product_id'].max()) # range from 1 to 49688
# top 20 product_ids by number of rows:
print(prior.groupby('product_id')['order_id'].count().reset_index().sort_values('order_id',ascending=False)[:20])


# **add_to_cart_order** -> this is simply the order the products were added to basket within an order

# In[ ]:


print(len(prior['add_to_cart_order'].unique())) # 145 unique add_to_cart_order values
print(prior['add_to_cart_order'].min(),prior['add_to_cart_order'].max()) # range from 1 to 145
# top 20 add_to_cart_order values by number of rows:
print(prior.groupby('add_to_cart_order')['order_id'].count().reset_index().sort_values('order_id',ascending=False)[:20])


# **reordered**

# In[ ]:


print(len(prior['reordered'].unique())) # 2 unique reordered values
print(prior['reordered'].min(),prior['reordered'].max()) # either a 0 or a 1 -> Boolean values
# count of rows by reordered value
print(prior.groupby('reordered')['order_id'].count().reset_index().sort_values('order_id',ascending=False))
# reordered = 1 -> 58.97% of rows
# reordered = 0 -> 41.03%


# ## Train Data ##

# In[ ]:


train[:15]


# In[ ]:


train.shape # 1,384,617 rows (32,434,489 in prior)


# **order_id**

# In[ ]:


print(len(train['order_id'].unique())) # 131,209 unique order_ids (3,214,874 in prior)
print(train['order_id'].min(),train['order_id'].max()) # range from 1 to 3421070
# top 20 order_ids by number of rows:
print(train.groupby('order_id')['product_id'].count().reset_index().sort_values('product_id',ascending=False)[:20])


# **product_id**

# In[ ]:


print(len(train['product_id'].unique())) # 39,123 unique product_ids (49,677 in prior)
print(train['product_id'].min(),train['product_id'].max()) # range from 1 to 49688
# top 20 product_ids by number of rows:
print(train.groupby('product_id')['order_id'].count().reset_index().sort_values('order_id',ascending=False)[:20])


# **add_to_cart_order**

# In[ ]:


print(len(train['add_to_cart_order'].unique())) # 80 unique add_to_cart_order values (145 in prior)
print(train['add_to_cart_order'].min(),train['add_to_cart_order'].max()) # range from 1 to 80
# top 20 add_to_cart_order values by number of rows:
print(train.groupby('add_to_cart_order')['order_id'].count().reset_index().sort_values('order_id',ascending=False)[:20])


# **reordered**

# In[ ]:


print(len(train['reordered'].unique())) # 2 unique reordered values
print(train['reordered'].min(),train['reordered'].max()) # either a 0 or a 1 -> Boolean values
# count of rows by reordered value
print(train.groupby('reordered')['order_id'].count().reset_index().sort_values('order_id',ascending=False))
# reordered = 1 -> 59.86% of rows (58.97% in prior)
# reordered = 0 -> 40.14% (41.03% in prior)


# ##Orders Data##

# In[ ]:


orders[:15]


# In[ ]:


orders.shape # 3,421,083 rows (there are 3,214,874 order_ids in prior)


# **order_id**

# In[ ]:


print(len(orders['order_id'].unique())) # 3,421,083 unique order_ids
print(orders['order_id'].min(),orders['order_id'].max()) # range from 1 to 3421083
# top 20 order_ids by number of rows:
print(orders.groupby('order_id')['user_id'].count().reset_index().sort_values('user_id',ascending=False)[:20])
# every order_id appears once -> as expected because it is a unique identifier


# **user_id**

# In[ ]:


print(len(orders['user_id'].unique())) # 206,209 unique user_ids
print(orders['user_id'].min(),orders['user_id'].max()) # range from 1 to 206209
# surjective mapping between order_id and user_id in this file
# top 20 user_ids by number of rows:
print(orders.groupby('user_id')['order_id'].count().reset_index().sort_values('order_id',ascending=False)[:20])


# **eval_set**

# In[ ]:


print(len(orders['eval_set'].unique())) # 3 unique eval_set values
print(orders['eval_set'].unique()) # prior, train, test
# number of rows in orders grouped by the 3 eval_set values
print(orders.groupby('eval_set')['order_id'].count().reset_index().sort_values('order_id',ascending=False))
# prior = 93.97% of rows
# train = 3.84%
# test = 2.19%


# **order_number** -> this seems to simply be the order in which a user makes orders

# In[ ]:


print(len(orders['order_number'].unique())) # 100 unique order_numbers
print(orders['order_number'].min(),orders['order_number'].max()) # range from 1 to 100
# top 20 order_numbers by number of rows:
print(orders.groupby('order_number')['order_id'].count().reset_index().sort_values('order_id',ascending=False)[:20])
# order_numbers 1-4 appear for all 206,209 order_ids


# **order_dow** -> day of week for order

# In[ ]:


print(len(orders['order_dow'].unique())) # 7 unique order_dow values, as expected
print(orders['order_dow'].min(),orders['order_dow'].max()) # range from 0 to 6
# count of rows grouped by day of week 
print(orders.groupby('order_dow')['order_id'].count().reset_index().sort_values('order_id',ascending=False))
# order_numbers 1-4 appear for all 206,209 order_ids


# In[ ]:


orders.groupby('order_dow')['order_id'].count().reset_index().sort_values('order_id',ascending=False).plot(kind='bar')


# **order_hour_of_day**

# In[ ]:


print(len(orders['order_hour_of_day'].unique())) # 24 unique order_hour_of_day values, as expected
print(orders['order_hour_of_day'].min(),orders['order_hour_of_day'].max()) # range from 0 to 23
# count of rows grouped by hour of day
print(orders.groupby('order_hour_of_day')['order_id'].count().reset_index().sort_values('order_id',ascending=False))


# In[ ]:


orders.groupby('order_hour_of_day')['order_id'].count().reset_index().sort_values('order_hour_of_day',ascending=True).plot(kind='bar')


# **days_since_prior_order**

# In[ ]:


print(len(orders['days_since_prior_order'].unique())) # 32 unique days_since_prior_order values
print(orders['days_since_prior_order'].min(),orders['days_since_prior_order'].max()) # range from 0 to 30
# also NULL values
# count of rows grouped by days_since_prior_order
print(orders.groupby('days_since_prior_order')['order_id'].count().reset_index().sort_values('order_id',ascending=False))
# NULL values
print(len(orders[orders['days_since_prior_order'].isnull()]))
# 206,209 rows with NULL values for this column


# In[ ]:


orders.groupby('days_since_prior_order')['order_id'].count().reset_index().sort_values('days_since_prior_order',ascending=True).plot(kind='bar')


# In[ ]:


# NULL values for days_since_prior_order only exist for prior data, as expected
orders[orders['days_since_prior_order'].isnull()]['eval_set'].unique()


# ##Products Data##

# In[ ]:


products[:15]


# In[ ]:


products.shape


# In[ ]:




