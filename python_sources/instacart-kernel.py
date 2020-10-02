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
idir = os.path.join('..', 'input')
# Any results you write to the current directory are saved as output.
df_aisles = pd.read_csv(os.path.join(idir, 'aisles.csv'))
print(df_aisles.head())


# In[ ]:


df_department = pd.read_csv(os.path.join(idir, 'departments.csv'))
df_opp = pd.read_csv(os.path.join(idir, 'order_products__prior.csv'))
df_opt = pd.read_csv(os.path.join(idir, 'order_products__train.csv'))
df_orders = pd.read_csv(os.path.join(idir, 'orders.csv'))
df_products = pd.read_csv(os.path.join(idir, 'products.csv'))


# In[ ]:


for df in [df_aisles, df_department, df_opp, df_opt, df_orders, df_products]:
    print(df.head())


# In[ ]:


for df in [df_aisles, df_department, df_opp, df_opt, df_orders, df_products]:
    print(df.columns)


# In[ ]:


df_orders.describe()


# In[ ]:


df_orders.head()


# In[ ]:


df_orders['eval_set'].value_counts()


# In[ ]:


df = pd.merge(
    left = df_products,
    right = df_department,
    on = 'department_id',
    left_index = False,
    right_index = False
)

df = pd.merge(
    left = df,
    right = df_aisles,
    on = 'aisle_id',
    left_index = False,
    right_index = False
)

df_prior = pd.merge(
    left = df,
    right = df_opp,
    on = 'product_id',
    left_index = False,
    right_index = False
)

df_prior = pd.merge(
    left = df_prior,
    right = df_orders[df_orders['eval_set'] == 'prior'],
    on = 'order_id',
    left_index = False,
    right_index = False
)

df_train = pd.merge(
    left = df,
    right = df_opt,
    on = 'product_id',
    left_index = False,
    right_index = False
)

df_train = pd.merge(
    left = df_train,
    right = df_orders[df_orders['eval_set'] == 'train'],
    on = 'order_id',
    left_index = False,
    right_index = False
)

df_test = df_orders.loc[df_orders['eval_set'] == 'test', :]
    


# In[ ]:


df_train.head()


# In[ ]:


df_prior['reordered'].value_counts()


# In[ ]:


df_prior.groupby('product_name')['reordered'].sum().sort_values(ascending=False)


# In[ ]:


# sum reorder by dept
# days since last order
df_prior.groupby('department')['reordered'].sum().sort_values(ascending=False)


# 

# TODO:
# - When training our model keep each customer separate
# - Cluster customers
# 

# In[ ]:





# In[ ]:





# In[ ]:





# Ideas to pursue:
# - Look at a single customer first (highest 5 customers - total number of orders, average number of items per order)
# - Graph the number of days since last order
# 

# In[ ]:


import seaborn as sns

df_prior['order_dow'].value_counts().sort_index().plot.bar()


# In[ ]:


df_user_order_cnts = df_prior.groupby(['user_id', 'order_id'])[['product_id']].count()
df_order_agg = df_user_order_cnts.reset_index().groupby(['user_id'])['product_id'].agg({'Avg Items Per Order':'mean', 'Number of Orders':'count'})
df_order_agg.sort_values(by='Number of Orders', ascending=False)


# In[ ]:


df_prior[df_prior['user_id'] == 152340].groupby(['product_id', 'product_name'])['aisle'].count().sort_values(ascending=False)


# In[ ]:


mask = (df_prior['user_id'] == 152340) & (df_prior['product_id'] == 35461)
df_prior.loc[mask, :].groupby('order_dow').count()['product_id'].plot.bar()


# In[ ]:


df_prior.loc[mask, :].groupby('days_since_prior_order').count()['product_id'].plot.bar()


# In[ ]:


sns.heatmap(df_prior.isnull(), cbar=False)


# In[ ]:


sns.heatmap(df_train.isnull(), cbar=False)


# In[ ]:


sns.heatmap(df_test.isnull(), cbar=False)

