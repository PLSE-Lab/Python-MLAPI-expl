#!/usr/bin/env python
# coding: utf-8

# **features geneartion**

# In[1]:


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


# In[2]:


orders_prior = pd.read_csv('../input/order_products__prior.csv' , dtype ={'order_id': np.int32,
                            'product_id': np.uint16,
                            'add_to_cart_order': np.int16,
                            'reordered': np.int8})


# In[3]:


orders_train = pd.read_csv('../input/order_products__train.csv' , dtype ={'order_id': np.int32,
                            'product_id': np.uint16,
                            'add_to_cart_order': np.int16,
                            'reordered': np.int8})


# In[4]:


orders = pd.read_csv('../input/orders.csv',dtype= {'order_id':np.int32,'user_id':np.int64,'evel_set':'category','order_number':np.int64,'order_dow':np.int8,'order_hours_of_day':np.int8,'days_dince_prior_order':np.float32 })


# In[5]:


orders.drop(orders.columns[[3,4,5]],axis = 1)
order_all = pd.concat([orders_prior,orders_train],axis = 0)
del orders_train
del orders_prior


# In[6]:


all_data = order_all.merge(orders,how='inner',on = 'order_id')


# In[7]:


orders_new = orders.drop(orders.columns[[3,4,5]],axis = 1)
del orders


# In[8]:


all_data = order_all.merge(orders_new,how='inner',on = 'order_id')


# In[9]:


all_data = all_data.sort_values(by='user_id')


# In[10]:


unique_products = all_data.groupby("product_id")["reordered"].aggregate({'total_reordered':'count','reorder_sum':sum}).reset_index()
unique_products['reorder_probability'] = unique_products['reorder_sum']/unique_products['total_reordered']                                                                                     
all_data = pd.merge(all_data,unique_products,how = 'inner',on = 'product_id')


# In[11]:


all_data = all_data.sort_values(by = 'user_id',ascending = True)
all_data.head()

 


# In[ ]:


temp = all_data.drop(all_data.columns[[0,2,3,5,6,7,8,9]],axis = 1)
temp.head()


# In[ ]:


tempnew=temp.groupby(['product_id','user_id'])["product_id"].aggregate({'purbyusers':'count'}).reset_index()
tempnew.head()


# In[ ]:


tempnew.user_id.max()
del temp


# In[ ]:


new_all_data=all_data.merge(tempnew,how='inner',on=['user_id','product_id'])

