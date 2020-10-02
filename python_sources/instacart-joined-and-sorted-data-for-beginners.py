#!/usr/bin/env python
# coding: utf-8

# ## For beginners, having trouble with huge amount of data files and just want to get the joined data, I have created this kernel to ease your pain.
# 
# **You can directly download the 'hdf' file  from the OUTPUT TAB  OR include it in your kernel and read it using**  `df=pd.read_hdf('final.hdf', key='final_df', mode='r')` 

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


# # Making one DataFrame of all the date.
# ### Let's have a look at the different files we have in our data and their respective columns.
# 
# **1. aisles.csv**:
# Shape: 134x2
# * `aisle_id`: aisle identifier
# * `aisle`: the name of the aisle
# 
# **2. department.csv**:
# Shape: 21x2
# * `department_id`: department identifier
# * `department`: the name of the department
# 
# **(3,4). order_products__SET.csv**:
# * `order_id`: foreign key
# * `product_id`: foreign key
# * `add_to_cart_order`: order in which each product was added to cart
# * `reordered`: 1 if this product has been ordered by this user in the past, 0 otherwise
# 
# where `SET` is one of the four following evaluation sets (`eval_set` in `orders`):
# * `"prior"`: orders prior to that users most recent order (~32.4mx4 )
# * `"train"`: training data supplied to participants (1.38mx4)
# 
# **5. orders.csv**:
# * `order_id`: order identifier
# * `user_id`: customer identifier
# * `eval_set`: which evaluation set this order belongs in (see `SET` described below)
# * `order_number`: the order sequence number for this user (1 = first, n = nth)
# * `order_dow`: the day of the week the order was placed on
# * `order_hour_of_day`: the hour of the day the order was placed on
# * `days_since_prior`: days since the last order, capped at 30 (with NAs for `order_number` = 1)
# 
# (source:https://gist.github.com/jeremystan/c3b39d947d9b88b3ccff3147dbcf6c6b/data_description.mds)

# In[ ]:


# importing 'orders.csv' and sorting it by 'user_id' and 'order_number'
orders=pd.read_csv('../input/orders.csv')
orders=orders[['user_id','order_number', 'order_id', 'eval_set', 'order_dow',
       'order_hour_of_day', 'days_since_prior_order']].sort_values(['user_id','order_number'])
#first order of user_id==1 can be seen as below:
print(orders.head())


# In[ ]:


#Examining the different SETs in the data.
print(orders['eval_set'].unique())
#We can drop the 'test' SET
orders.drop(orders[orders['eval_set']=='test'].index,inplace=True)
print(orders['eval_set'].unique())


# In[ ]:


#We need to map all the orders based on 'order_id' and 'eval_set'. But before we need to import both SETs of 'order_products_*.csv'
order_prior=pd.read_csv('../input/order_products__prior.csv')
print(order_prior.head())


# In[ ]:


order_train=pd.read_csv('../input/order_products__train.csv')
print(order_train.head())


# In[ ]:


p_orders=orders[orders['eval_set']=='prior'].drop('eval_set',axis=1)
df_prior=pd.merge(p_orders,order_prior,left_on='order_id',right_on='order_id')
df_prior['eval_set']='prior'
df_prior.head()


# In[ ]:


t_orders=orders[orders['eval_set']=='train'].drop('eval_set',axis=1)
df_train=pd.merge(t_orders,order_train,left_on='order_id',right_on='order_id')
df_train['eval_set']='test'
df_train.head()


# In[ ]:


df=pd.concat([df_prior,df_train],ignore_index=True)
df=df.sort_values(['user_id','order_number','add_to_cart_order'])
df.head()


# In[ ]:


#deleting the dataframes that arent needed
del orders
del order_prior
del order_train
del p_orders
del t_orders
del df_prior
del df_train


# In[ ]:


#importing 'products.csv', 'aisles.csv' and 'department.csv'
products=pd.read_csv('../input/products.csv')
aisles=pd.read_csv('../input/aisles.csv')
departments=pd.read_csv('../input/departments.csv')


# In[ ]:


#Merging products and aisles
products_aisles_df= pd.merge(products,aisles,left_on='aisle_id',right_on='aisle_id').sort_values('product_id')
print(products_aisles_df.head())


# In[ ]:


#Merging products_aisles_df with departments to get 'products_df'
products_df=pd.merge(products_aisles_df,departments,left_on='department_id',right_on='department_id')
print(products_df.head())


# In[ ]:


#deleting the dataframes not needed
del products
del aisles
del departments
del products_aisles_df


# In[ ]:


final_df=pd.merge(df,products_df,left_on='product_id',right_on='product_id')
final_df=final_df.sort_values(['user_id','order_number','add_to_cart_order'])
final_df.head(20)


# In[ ]:


final_df.to_hdf('final.hdf','final_df',mode='w',Table=True)


# In[ ]:





# In[ ]:




