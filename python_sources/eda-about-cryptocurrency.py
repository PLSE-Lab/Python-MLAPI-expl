#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
df = pd.read_csv("../input/CryptocoinsHistoricalPrices.csv", index_col = 0)
df.head()


# In[2]:


df.dtypes


# 1. Create a faceted plot in ggplot of a public data set use size, shape and color as well as facets.

# In[3]:


from pandas import Series
from pandas import TimeGrouper


# In[4]:


df.isnull().sum


# In[5]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


from ggplot import *


# In[7]:


df['Date'] =  pd.to_datetime(df['Date'], format='%Y-%m-%d')
df.head()


# In[8]:


print(type(df))


# In[9]:


df = df.dropna(axis = 0)
df.head()


# In[10]:


plt.subplots(figsize=(20,15))
ax = plt.axes()
ax.set_title("Correlation of cryptocurrency price")
corr = df.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)


# In[11]:


df = df[(df.coin == 'BTC') | (df.coin == 'ETH') | (df.coin == 'XRP') | (df.coin == 'BCH') | (df.coin == 'XEM') | (df.coin == 'LTC')]


# In[12]:


df.head()


# In[13]:


p = ggplot(df, aes(x='Date',y='Close')) + geom_line(color = 'steelblue') +    scale_x_date(breaks=date_breaks('3 months'), labels = '%Y-%m') +     stat_smooth(color='blue') + facet_wrap("coin") +     theme_bw() +     labs(title = "Price of Cryptocurrent",
        x = 'Time',
        y = 'Price') + \
    theme(axis_title_x = element_text(color = 'gray', size=16, family="Arial"),
         axis_title_y = element_text(color = 'gray', size=16, family="Arial")) 
    
p
    


# Not every kinds of cryptocurrency goes crasy. 

# In[14]:


p = ggplot(df, aes(x='Date',y='Close', color='coin')) + geom_point() + geom_line() + stat_smooth(color='blue')
p


# 2.Create a Correlation Heatmap in Seaborn using a public dataset.
# 

# In[15]:


plt.subplots(figsize=(20,15))
ax = plt.axes()
ax.set_title("Correlation of cryptocurrency price")
corr = df.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)


# There is no significant correlation between the open price with close/ high / low price of cryptocurrency.

# 3. Create your own Test and Training sets using a public dataset.

# In[16]:


from sklearn.model_selection import train_test_split
X, Y = df.iloc[:, 1:].values, df.iloc[:, 0].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                    test_size=0.4,
                    random_state=0)


# In[ ]:




