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


df_aisles = pd.read_csv("../input/aisles.csv")
df_departments = pd.read_csv("../input/departments.csv")
df_order_products__prior = pd.read_csv("../input/order_products__prior.csv")
df_order_products__train = pd.read_csv("../input/order_products__train.csv")
df_orders = pd.read_csv("../input/orders.csv")
df_products = pd.read_csv("../input/products.csv")


# In[ ]:


df_list = [df_aisles,df_departments,df_order_products__prior,df_order_products__train,df_orders,df_products]
for i in df_list:
    print(i.shape)


# In[ ]:


df_aisles.head(20)


# In[ ]:


df_aisles.shape


# In[ ]:


df_departments


# In[ ]:


df_departments.shape


# In[ ]:


df_order_products__prior.head()


# In[ ]:


df_order_products__prior.shape


# In[ ]:


df_order_products__train.head()


# In[ ]:


df_order_products__train.shape


# In[ ]:


df_orders.head()


# In[ ]:


df_orders.shape


# In[ ]:


df_products.head()


# In[ ]:


df_products.shape


# In[ ]:




