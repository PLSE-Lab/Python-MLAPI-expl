#!/usr/bin/env python
# coding: utf-8

# ## Let's take a look at the contents of the data for the time being!

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from IPython.display import display

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# In[ ]:


aisles = pd.read_csv('../input/aisles.csv')
departments = pd.read_csv('../input/departments.csv')
order_products__prior = pd.read_csv('../input/order_products__prior.csv')
order_products__train = pd.read_csv('../input/order_products__train.csv')
order = pd.read_csv('../input/orders.csv')
products = pd.read_csv('../input/products.csv')
sample_submission = pd.read_csv('../input/sample_submission.csv')


# # order.csv

# In[ ]:


display(order)
print(order.shape)


# In[ ]:


order.groupby('eval_set').size()


# There seems to be prior in addition to train and test.
# Looking at order.csv, it seems that we have made multiple orders per person,
# It seems like a task to predict the contents of the last order (train or test).
# 
# The order order is written in order_number, and the number of days elapsed since the previous order is indicated by days_since_prior_order.

# In[ ]:


order.groupby('user_id').size().hist()
print(order.groupby('user_id').size().mean())


# It seems that they make an average of 16.5 orders per person.
# The data seems to be the maximum limit of 100 times.

# # order_products__*.csv

# In[ ]:


order_products__prior


# It seems that there are many patterns that buy multiple items in one order. Add_to_cart_order is the order in which you put it in cart, reordered indicates whether you have bought the item before.

# In[ ]:


order_products__prior.groupby('order_id').size().hist()
print(order_products__prior.groupby('order_id').size().mean())


# It seems to buy ten items on average on an order.

# In[ ]:


order_products__train.head()


# Train is the same as prior.

# # products.csv

# In[ ]:


display(products)
print(products.shape)


# It seems that products.csv represents detailed contents of the product as its name suggests.
# There seems to be 49688 kinds of products in all.
# Details of aisle_id, department_id are below.

# # aisles.csv, departments.csv

# In[ ]:


display(aisles.head())
print(aisles.shape)
display(departments.head())
print(departments.shape)


# Aisle seems to be a passage (?) In the shop where the merchandise is located and department indicates the genre of the merchandise. I still do not really understand.

# # sample_submission.csv

# In[ ]:


display(sample_submission.head())
print(sample_submission.shape)
display(sample_submission.ix[0,'products'])


# Fill out and submit the products that you think will be purchased at test order order_id. Let's do our best!
