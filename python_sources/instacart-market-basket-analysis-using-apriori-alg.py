#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

#import numpy as np # linear algebra
#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori,association_rules
import zipfile
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# Importing files in script using **zipfile** library(csv's are zipped) and pandas to read file.
# 
# *Note: Didn't use zip compression in read_csv as zip file contains more than one file.*

# In[ ]:


zf1 = zipfile.ZipFile('../input/instacart-market-basket-analysis/order_products__prior.csv.zip')
order_prod = pd.read_csv(zf1.open('order_products__prior.csv'))
zf2 = zipfile.ZipFile('../input/instacart-market-basket-analysis/products.csv.zip')
prod = pd.read_csv(zf2.open('products.csv'))


# Merging the files to get a single dataframe with both order and item details

# In[ ]:


data = pd.merge(order_prod,prod,how='inner',on='product_id')
data.head()


# Sampling data of first **10000 order** transactions from total population, due to memory constraint.

# In[ ]:


data = data[data['order_id']<10000]


# In[ ]:


df_item = data[['order_id','product_name']].copy()
df_item.rename(columns={'order_id':'order','product_name':'items'},inplace=True)
df_item['temp']=1


# Making transaction data ready for Frequent item set mining.

# In[ ]:


df = df_item.groupby(['order','items'])['temp'].sum().unstack().fillna(0)


# An encoder to covert numbers greater than 1 to 1 and less than 0 to 0.

# In[ ]:


def myencoder(i):
    if i <= 0:
        return 0
    elif i>=1:
        return 1


# In[ ]:


df.applymap(myencoder)


# **Frequent Itemsets Mining**

# In[ ]:


freq_itemsets = apriori(df,min_support=0.01,use_colnames=True)
freq_itemsets


#  **Association Rules Mining**

# In[ ]:


rules = association_rules(freq_itemsets,metric='lift',min_threshold=1)
rules.sort_values(by='confidence',ascending=False)


# Filtering rules based on some of my own assumptions.
# 
# Note: The chosen threshold for confidence and lift is entirely upto my trial and test. No technical way used.

# In[ ]:


rules[(rules['confidence']>0.16) & (rules['lift']>2)]

