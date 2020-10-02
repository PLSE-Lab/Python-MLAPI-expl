#!/usr/bin/env python
# coding: utf-8

# 

# In[ ]:


import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.offsetbox import AnchoredText
import seaborn as sns
plt.style.use('ggplot')
from IPython.display import display


# In[ ]:


data_set = pd.read_csv('../input/products.csv') #import data set and take a look
data_set.head(100)


# In[ ]:


data_set['aisle_id'].value_counts() #count how many products are purchased by aisle
aisle_names = pd.read_csv('../input/aisles.csv') #import aisle names for future use maybe


# In[ ]:


data_set.sort_values(by=['aisle_id']) #sort data by aisle_id


# In[ ]:


data_set_aisle = data_set.groupby('aisle_id').count() #count entries per aisle


# In[ ]:


#create a count column from the 'groupby' table
data_set_aisle.reset_index()

data_set_aisle.drop('product_name', axis=1, inplace= True)
data_set_aisle.drop('department_id', axis=1, inplace= True)
#drop these columns because they are redundant, and 
#replace with with a 'count' column
data_set_aisle = data_set_aisle.rename(columns={'product_id': 'count'})

data = data_set_aisle.reset_index() #save the grouped list to data

#import the dataset with the names of the aisle and merge 
aisle_names = pd.read_csv('../input/aisles.csv')
data['aisle']= aisle_names['aisle']


# In[ ]:


data


# In[ ]:




plt.figure(figsize=(10,10))
plt.bar(data.index, data['count'])
plt.xticks(data.index, data['count'])
plt.ylabel('Count')
plt.xlabel('Aisle')
plt.tight_layout()
plt.show()


# In[ ]:


orders_prior = pd.read_csv('../input/order_products__prior.csv')
orders_train = pd.read_csv('../input/order_products__train.csv')


# In[ ]:


orders_prior.head(10)


# In[ ]:


orders_train.head(10)


# In[ ]:


orders_prior.reordered.sum()/orders_prior.shape[0]


# In[ ]:


orders_train.reordered.sum()/orders_train.shape[0]


# <p>About 59% of products were reordered.</p>

# In[ ]:


grouped_prior = orders_prior.groupby("order_id")["reordered"].aggregate("sum").reset_index()
grouped_prior["reordered"].loc[grouped_prior["reordered"]>1]=1
grouped_prior.reordered.value_counts()/grouped_prior.shape[0]


# In[ ]:


grouped_prior = orders_train.groupby("order_id")["reordered"].aggregate("sum").reset_index()
grouped_prior["reordered"].loc[grouped_prior["reordered"]>1]=1
grouped_prior.reordered.value_counts()/grouped_prior.shape[0]


# About 12% of prior data products were not reordered, and 0.6% of train products were not reordered. 

# In[ ]:


grouped_prior = orders_train.groupby("order_id")["add_to_cart_order"].aggregate("max").reset_index()
counts = grouped_prior.add_to_cart_order.value_counts()

plt.figure(figsize=(12,8))
sns.barplot(counts.index, counts.values, alpha=0.8)
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Number of products in the given order', fontsize=12)
plt.xticks(rotation='vertical')
plt.show()


# <p>We can see that 5 products in an order is the most common order size. </p>

# In[ ]:


data_set.head()


# In[ ]:


aisle_names.head()


# In[ ]:


departments = pd.read_csv('../input/departments.csv')
departments.head()


# In[ ]:


orders_prior = pd.merge(orders_prior, data_set, on='product_id', how='left')
orders_prior = pd.merge(orders_prior, aisle_names, on='aisle_id', how='left')
orders_prior = pd.merge(orders_prior, departments, on = 'department_id', how='left')#just like in sql

orders_prior.head()


# In[ ]:


count = orders_prior['product_name'].value_counts().reset_index().head(20)
count.columns = ['product_name','frequency_county']
count


# In[ ]:


count = orders_prior['aisle'].value_counts().head(20)
plt.figure(20,15)
sns.barplot(count.index,count.values,alpha=0.8)
plt.ylabel('Occurences')
plt.xlabel('Aisle')
plt.show()

