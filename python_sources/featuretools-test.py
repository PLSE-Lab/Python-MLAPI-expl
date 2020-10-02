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


# https://docs.featuretools.com/#minute-quick-start

# In[1]:


import featuretools as ft


# In[3]:


data = ft.demo.load_mock_customer()
customers_df = data["customers"]


# In[4]:


customers_df


# In[6]:


sessions_df = data["sessions"]
sessions_df.sample(5)


# In[8]:


transactions_df = data["transactions"]
transactions_df.sample(5)


# In[9]:


entities = {
   "customers" : (customers_df, "customer_id"),
  "sessions" : (sessions_df, "session_id", "session_start"),
   "transactions" : (transactions_df, "transaction_id", "transaction_time")
}


# In[11]:


relationships = [("sessions", "session_id", "transactions", "session_id"),
                 ("customers", "customer_id", "sessions", "customer_id")]


# In[12]:


feature_matrix_customers, features_defs = ft.dfs(entities=entities,
                                                  relationships=relationships,
                                                 target_entity="customers")


# In[13]:


feature_matrix_customers


# In[14]:


feature_matrix_sessions, features_defs = ft.dfs(entities=entities,
                                                 relationships=relationships,
                                                 target_entity="sessions")


# In[15]:


feature_matrix_sessions, features_defs = ft.dfs(entities=entities,
                                                 relationships=relationships,
                                                 target_entity="sessions")


# https://docs.featuretools.com/automated_feature_engineering/afe.html

# In[17]:


es = ft.demo.load_mock_customer(return_entityset=True)
es


# In[19]:


feature_matrix, feature_defs = ft.dfs(entityset=es,
                                       target_entity="customers",
                                       agg_primitives=["count"],
                                       trans_primitives=["month"],
                                       max_depth=1)
feature_matrix


# In[21]:


feature_matrix, feature_defs = ft.dfs(entityset=es,
                                       target_entity="customers",
                                       agg_primitives=["mean", "sum", "mode"],
                                       trans_primitives=["month", "hour"],
                                       max_depth=2)
feature_matrix


# In[22]:


feature_matrix[['MODE(sessions.HOUR(session_start))']]


# In[30]:


feature_matrix, feature_defs = ft.dfs(entityset=es,
                                       target_entity="sessions",
                                       agg_primitives=["mean", "sum", "mode"],
                                       trans_primitives=["month", "hour"],
                                       max_depth=2)


# In[27]:


feature_matrix.head(5)


# In[28]:


feature_matrix[['customers.MEAN(transactions.amount)']].head(5)


# In[ ]:




