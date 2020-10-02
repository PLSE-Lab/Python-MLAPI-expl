#!/usr/bin/env python
# coding: utf-8

# # Instacart Market Basket Analysis
# The dataset for this competition is a relational set of files describing customers' orders over time. The goal of the competition is to predict which products will be in a user's next order.
# 
# The dataset consists of information about 3.4 million grocery orders, distributed across 6 csv files.
# 
# Here is a first ensight of the competition dataset.
# Importing the required libraries

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data plotting
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Reading all inputs:
# There are 6 input files to be read

# In[ ]:


aisle_df = pd.read_csv('../input/aisles.csv')
departments_df = pd.read_csv('../input/departments.csv')
prior_df = pd.read_csv('../input/order_products__prior.csv')
train_df = pd.read_csv('../input/order_products__train.csv')
orders_df = pd.read_csv('../input/orders.csv')
products_df = pd.read_csv('../input/products.csv')


# re-ordered or not vs add to cart sequence

# In[ ]:


train_df.head()
temp = train_df.groupby(['add_to_cart_order','reordered'])
temp=temp.size().unstack()
temp =temp.head(30)
temp.plot(kind = "bar", rot = 45, stacked = False )


# from above its clear that if  the probability of re-ordering for first 20 items added to cart is higher.
# Checking the most re-ordered product

# In[ ]:


temp = prior_df[prior_df.reordered == 1]
temp = temp.product_id.value_counts()
temp = temp.head(30)
temp.plot(kind = "bar", rot = 45, stacked = False)


# 

# 
