#!/usr/bin/env python
# coding: utf-8

# #Take Up  Market Basket

# ##Import Libs

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='whitegrid', context='notebook', palette='Set2')
sns.despine()

import os, gc, psutil
process = psutil.Process(os.getpid())
import warnings
warnings.filterwarnings('ignore')


# ##Load Data

# In[2]:



#load inputFiles

depart = pd.read_csv('../input/departments.csv', engine='c',
							 dtype={'department_id':np.int8})   
aisles = pd.read_csv('../input/aisles.csv', engine='c',
						dtype={'aisle_id':np.uint8})
products = pd.read_csv('../input/products.csv', engine='c',
                       	  dtype={'product_id':np.uint16,
                                 'aisle_id':np.uint8,
                                 'department_id':np.int8})
orders = pd.read_csv('../input/orders.csv', engine='c',
                        dtype={'order_id':np.int32,
                               'user_id':np.int32,
                               'order_number':np.int8,
                               'order_dow':np.int8,
                               'order_hour_of_day':np.int8})
op_train = pd.read_csv('../input/order_products__train.csv', engine='c',
                          dtype={'order_id':np.int32, 
                                 'product_id':np.uint16, 
                                 'add_to_cart_order':np.int8, 
                                 'reordered':np.int8})
op_prior = pd.read_csv('../input/order_products__prior.csv', engine='c',
                          dtype={'order_id':np.int32, 
                                 'product_id':np.uint16, 
                                 'add_to_cart_order':np.int8, 
                                 'reordered':np.int8})


print('Memory usage ~ {:.2f} GB'.format(process.memory_info().rss / (float(1024**3))))


# ##Overview Data

# In[ ]:


depart.info()
print('-'*40)
aisles.info()
print('-'*40)
products.info()
print('-'*40)
orders.info()
print('-'*40)
op_train.info()
print('-'*40)
op_prior.info()


# In[ ]:


orders.isnull().sum()


# In[ ]:


orders[orders['days_since_prior_order'].isnull()]


# In[ ]:


op_prior.isnull().sum()


# In[7]:


print('Memory usage ~ {:.2f} GB'.format(process.memory_info().rss / (float(1024**3))))


# In[ ]:


orders.describe()


# In[ ]:


depart.head()


# In[ ]:


aisles.head()


# In[ ]:


products.head()


# In[ ]:


orders.head()


# In[ ]:


op_train.head()


# In[ ]:


op_prior.head()


# In[ ]:


gc.collect()


# ##Merge orders to orders_train, orders_test, op_prior

# In[ ]:


orders_test = orders[orders['eval_set']=='test']
orders_test.info()


# In[ ]:


orders_train = orders[orders['eval_set']=='train']


# In[ ]:


op_train = pd.merge(op_train, orders, on='order_id')
op_train.info()


# In[ ]:


op_prior = pd.merge(op_prior, orders, on='order_id')
op_prior.info()


# In[ ]:


del orders


# In[ ]:


gc.collect()
print('Memory usage ~ {:.2f} GB'.format(process.memory_info().rss / (float(1024**3))))


# ##Separate op_prior into train and test by user type
# 
# - op_prior_train as train_X
# - op_prior_test as test_X

# In[ ]:


op_prior_train = op_prior[op_prior['user_id'].isin(orders_train['user_id'])]
op_prior_train.info()


# ###Train_y

# In[ ]:


train_y = pd.merge(op_prior_train[['user_id', 'product_id']], op_train, how='left', on=['user_id', 'product_id'])['reordered']
train_y = train_y.fillna(0).astype(np.int8)


# In[ ]:


gc.collect()
print('Memory usage ~ {:.2f} GB'.format(process.memory_info().rss / (float(1024**3))))


# In[ ]:


train_X = pd.merge(op_prior_train, orders_train, on='user_id', suffixes=['_P', '_T'])
train_X.info()


# In[ ]:


op_prior_test = op_prior[op_prior['user_id'].isin(orders_test['user_id'])]
op_prior_test.info()


# In[ ]:


test_X = pd.merge(op_prior_test, orders_test, on='user_id', suffixes=['_P', '_T'])
test_X.info()


# In[ ]:


gc.collect()
print('Memory usage ~ {:.2f} GB'.format(process.memory_info().rss / (float(1024**3))))


# ##train_X 

# In[ ]:


train_X['y'] = pd.Series(train_y)


# In[ ]:


train_X.info()


# In[ ]:


train_X


# In[ ]:


gc.collect()
print('Memory usage ~ {:.2f} GB'.format(process.memory_info().rss / (float(1024**3))))


# In[ ]:


train_X.isnull().sum()


# In[ ]:


gc.collect()
print('Memory usage ~ {:.2f} GB'.format(process.memory_info().rss / (float(1024**3))))


# In[ ]:


train_X = pd.merge(train_X, products, how='left', on='product_id')

