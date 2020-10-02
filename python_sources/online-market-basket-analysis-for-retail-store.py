#!/usr/bin/env python
# coding: utf-8

# # This note is work in progress
# ** Online Market Basket Analysis**
# This data belongs to one of the real time big super store stratified sampling is taken so that 
# data is representative of the actual population to remove the bais and variance while making the prediction.

# ## Lets code, and bring the importand libraries 

# In[ ]:


import os
import re
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import mlxtend
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
sns.set()


# ### Import the sampled data for online retail store and visualize the result set

# In[ ]:


print(os.listdir("../input"))


# In[ ]:


retail_data = pd.read_excel('../input/Online Retail.xlsx',sep='delimiter')


# In[ ]:


retail_data.head(30)


# ### Total Number product sold

# In[ ]:


retail_data.info()


# In[ ]:


retail_data.describe()


# In[ ]:


len(retail_data)


# ### Total Number of unique items present in the datasets

# In[ ]:


retail_data['Description'].unique().shape


# ### Total number of customers come to store

# In[ ]:


retail_data['CustomerID'].unique().shape


# In[ ]:


(retail_data['Country'].unique())


# In[ ]:


retail_data.columns


# In[ ]:


retail_data.loc[retail_data['Quantity'].argmax()]


# In[ ]:





# In[ ]:


retail_data['Description'] = retail_data['Description'].str.strip()


# In[ ]:


retail_data['Description']


# In[ ]:


basket = (retail_data[retail_data['Country'] =="France"]
          .groupby(['InvoiceNo', 'Description'])['Quantity']
          .sum().unstack().reset_index().fillna(0)
          .set_index('InvoiceNo'))


# In[ ]:


basket.head()


# In[ ]:


def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1
basket_sets = basket.applymap(encode_units)
basket_sets.drop('POSTAGE', inplace=True, axis=1)


# In[ ]:


frequent_itemsets = apriori(basket_sets, min_support=0.07, use_colnames=True)
frequent_itemsets


# In[ ]:


rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules


# In[ ]:




