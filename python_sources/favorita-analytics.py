#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv("../input/train.csv", nrows=6000000, parse_dates=['date'],index_col='id')
train.tail(10)


# In[ ]:


df = train['unit_sales'].groupby(train['item_nbr']).count()
df = df.sort_values()
df_highest = df.nlargest(n=10)
df_highest.plot(kind='bar',figsize = (10,10),  title = "Top 10 items sold across all stores")
plt.show()


# In[ ]:


#Next we find lowest/less demand product. We use nsmallest to find the bottom 10 items,
#probably it doesn;t matter even if we stock them.
df_lowest = df.nsmallest(n=10)
df_lowest.plot(kind='bar',figsize = (10,10),  title = "Bottom 10 items sold")
plt.show()


# In[ ]:


items = pd.read_csv("../input/items.csv")
print (items.shape)
print (items.describe())


# In[ ]:


print (train.describe())


# In[ ]:


transaction = pd.read_csv("../input/transactions.csv")
print (transaction.describe())


# In[ ]:


stores = pd.read_csv("../input/stores.csv")
print (stores.head())


# In[ ]:


#Lets find out number of cities in each state, which in nothing but finding out number of stores in each
#in each state.
df = stores['city'].groupby(stores['state']).count()
df.plot(kind='bar', figsize = (12,8), yticks=np.arange(min(df), max(df)+1, 1.0), title = "Number of cities in each state")
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:


#Looks like onpromotion field is always NaN, if so we will get rid of that columns 
#from the training data
print(train['onpromotion'].notnull().any())
train_new=train.drop('onpromotion',axis=1)
print(train_new.tail(5))


# In[ ]:


test = pd.read_csv("../input/test.csv")


# In[ ]:


Y = train_new['item_nbr']


# In[ ]:


train_test = train.drop('item_nbr'))

