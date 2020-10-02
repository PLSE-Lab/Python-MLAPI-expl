#!/usr/bin/env python
# coding: utf-8

# ## Downsample data for fast experiments
# * Note that downsampling non randomly will bias results strongly. 
# * Will want to add W2V embeddings - https://www.kaggle.com/omarito/word2vec-for-products-analysis-0-01-lb

# In[ ]:


import os
print(os.listdir("../input"))

import numpy as np
import pandas as pd
# import lightgbm as lgb

IDIR = '../input/'


# In[ ]:


print('loading prior')
priors = pd.read_csv(IDIR + 'order_products__prior.csv', dtype={
            'order_id': np.int32,
            'product_id': np.uint16,
            'add_to_cart_order': np.int16,
            'reordered': np.int8})

print('loading train')
train = pd.read_csv(IDIR + 'order_products__train.csv', dtype={
            'order_id': np.int32,
            'product_id': np.uint16,
            'add_to_cart_order': np.int16,
            'reordered': np.int8})

print('loading orders')
orders = pd.read_csv(IDIR + 'orders.csv', dtype={
        'order_id': np.int32,
        'user_id': np.int32,
        'eval_set': 'category',
        'order_number': np.int16,
        'order_dow': np.int8,
        'order_hour_of_day': np.int8,
        'days_since_prior_order': np.float32})

print('loading products')
products = pd.read_csv(IDIR + 'products.csv', dtype={
        'product_id': np.uint16,
        'order_id': np.int32,
        'aisle_id': np.uint8,
        'department_id': np.uint8},
#         usecols=['product_id', 'aisle_id', 'department_id']
                      )

print('priors {}: {}'.format(priors.shape, ', '.join(priors.columns)))
print('orders {}: {}'.format(orders.shape, ', '.join(orders.columns)))
print('train {}: {}'.format(train.shape, ', '.join(train.columns)))


# In[ ]:


print("Original train mean reorder [no grouping]:", train.reordered.mean())
print("Original train mean reorder [order grouping]:", train.groupby("order_id")["reordered"].mean().mean())
print("Original train mean reorder [product grouping]:", train.groupby("product_id")["reordered"].mean().mean())


# In[ ]:


orders.nunique()


# In[ ]:


orders["user_count"] = orders.groupby("user_id")["order_id"].transform("count")


# In[ ]:


# orders.user_count.describe()
orders.user_count.hist()


# * Filter the very frequent and very "small" users
# * Leaves us with ~ 30%
# * This will likely make any local CV results irreperesentative of real evaluation! 

# In[ ]:


orders = orders.loc[(orders.user_count > 10) ] #  & (orders.user_count <96)
orders.shape


# In[ ]:


orders.user_count.hist()


# In[ ]:


orders.nunique()


# * Randomly downsample by user

# In[ ]:


import random

sample_users = random.sample(set(orders.user_id),50123)
len(sample_users)                 


# In[ ]:


orders = orders.loc[orders.user_id.isin(sample_users)]
orders.shape


# In[ ]:


orders.head()


# In[ ]:


train.head()


# In[ ]:


train.shape


# In[ ]:


train.loc[train.order_id.isin(orders.order_id)].shape


# In[ ]:


train = train.loc[train.order_id.isin(orders.order_id)]
print(train.shape)


# In[ ]:


train["product_count"] = train.groupby("product_id")["reordered"].transform("count")
train["product_count"].hist()


# In[ ]:


train["product_count"].describe()


# In[ ]:


print("Group filt train mean reorder [no grouping]:", train.reordered.mean())
print("Group filt train mean reorder [order grouping]:", train.groupby("order_id")["reordered"].mean().mean())
print("Group filt train mean reorder [product grouping]:", train.groupby("product_id")["reordered"].mean().mean())


# ## Filter low count products (i.e very rare)

# In[ ]:


train = train.loc[train["product_count"]>2]
print(train.shape)


# In[ ]:


train["product_count"].describe()


# In[ ]:


products.shape


# In[ ]:


print("Original train mean reorder [no grouping]:", train.reordered.mean())
print("Original train mean reorder [order grouping]:", train.groupby("order_id")["reordered"].mean().mean())
print("Original train mean reorder [product grouping]:", train.groupby("product_id")["reordered"].mean().mean())


# #### recalculate count feature without "external knowledge"

# In[ ]:


train["product_count"] = train.groupby("product_id")["reordered"].transform("count")
orders["user_count"] = orders.groupby("user_id")["order_id"].transform("count")


# ## Prior orders
# * 32 million rows~ : larger data!
# * Historical data
# * Filter by orders we have left (after user level filtering)

# In[ ]:


df_prior = pd.read_csv(IDIR + 'order_products__prior.csv', dtype={
            'order_id': np.int32,
            'product_id': np.uint16,
            'add_to_cart_order': np.int16,
            'reordered': np.int8})
print(df_prior.shape)
df_prior.head()


# In[ ]:


df_prior = df_prior.loc[df_prior.order_id.isin(orders.order_id)]
print(df_prior.shape)


# In[ ]:


df_prior.reordered.mean()


# In[ ]:


## Filter by items also
df_prior = df_prior.loc[df_prior.product_id.isin(train.product_id)]
print(df_prior.shape)
df_prior.head()


# In[ ]:


df_prior.reordered.mean().round(2)


# ## Merge product metadata tables
# * No reason not to have them in one table
# 

# In[ ]:


df_prod = pd.read_csv(IDIR + 'products.csv')
print(df_prod.shape)

df_prod = df_prod.merge(pd.read_csv(IDIR + 'aisles.csv'),on="aisle_id")
df_prod = df_prod.merge(pd.read_csv(IDIR + 'departments.csv'),on="department_id")

print(df_prod.shape)
df_prod.head()


# ## Export data
# * Don't forget to get the metadata seperately! 

# In[ ]:


train.to_csv("instacart_train_sample_50k.csv.gz",index=False,compression="gzip")
orders.to_csv("instacart_orders_sample_50k.csv.gz",index=False,compression="gzip")
df_prior.to_csv("instacart_priorOrders_sample_50k.csv.gz",index=False,compression="gzip")

df_prod.to_csv("products.csv.gz",index=False,compression="gzip")


# In[ ]:




