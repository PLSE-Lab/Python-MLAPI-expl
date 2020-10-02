#!/usr/bin/env python
# coding: utf-8

# # Ramblings and rummaging
# Let's explore the training data set to get an idea of what we're dealing with.
# 
# As we know, the data set comes from a real retail store. We are tasked with predicting the quantity of items purchased.

# In[10]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

# Where the competition data ends up after adding the hawaiiml data source
FILE_DIR = '../input/hawaiiml-data'
print(os.listdir(FILE_DIR))


# In[11]:


train = pd.read_csv(f'{FILE_DIR}/train.csv', encoding='ISO-8859-1')
train.head()


# In[ ]:


# basic pd info() doesn't reveal much about data values
train.info()


# ## What does the data represent?
# We can look for unique values for fields other than data identifier (id), time, quantity, and unit price. Assuming that stock id is uniquely paired with description, we can probably ignore description for now.
# 
# Let's see how many unique values we have for invoice id, customer id, stock id, and purchase date. Then, How much stuff do customers buy?
# 
# After that, we probably want to see how customers are related to country. For example, does every customer have only one country? How much stuff do customers buy?
# 
# ## How many unique values do we have for invoice id, customer id, stock id, and purchase date?|

# In[ ]:


for column in ['invoice_id', 'customer_id', 'stock_id', 'date']:
    print('%-12s  %8.2d' % (column, len(train[column].unique())))


# ## How many purchases (invoice ids) did each customer make?
# Hints from: https://www.shanelynn.ie/summarising-aggregation-and-grouping-data-in-python-pandas/

# In[ ]:


train[['invoice_id', 'customer_id']].groupby('customer_id')['invoice_id'].count().reset_index()


# Huh? That's kind of odd that there are a bunch that only have one purchase over time; assuming that the training data set captures a range of dates.
# 
# ## What range of dates are included in the training data?

# In[ ]:


train['date'].sort_values().unique()[[0,1,2,-2,-1]]


# ## Does every customer have one country?

# In[ ]:


train[['customer_id','country']].groupby('customer_id')['country'].unique()


# ## Why does customer id, 7, have more than one country?

# In[ ]:


train[train['customer_id'] == 7]['country'].unique()


# ## What does country mean?
# Is it the country of the customer or the destination of the purchase?
# 
# Let's look at the customer 7's purchases for 'eire'. It appears that multiple items are ordered on a single invoice.

# In[ ]:


train[['customer_id','invoice_id','country']][((train['customer_id'] == 7) & (train['country'] == 'eire'))].groupby('invoice_id').count()


# ## Let's look at customer 7's orders
# Select specific columns for "customer_id == 7" and then count the number of items for each invoice.

# In[ ]:


train[['customer_id','invoice_id','country']][(train['customer_id'] == 7)].groupby('invoice_id').count()


# Let's look at invoice 7 for customer 7 and sort the items ordered on that invoice by stock id.

# In[ ]:


train[['customer_id','invoice_id','date','time','stock_id','country']][((train['customer_id'] == 7) & (train['invoice_id'] == 7))].sort_values('stock_id')


# In[ ]:




