#!/usr/bin/env python
# coding: utf-8

# In[20]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sb


# In[21]:


data = pd.read_csv('../input/Mall_Customers.csv')
data.info()


# In[22]:


data.shape


# In[23]:


data.head()


# In[24]:


data.corr()


# In[25]:


f, ax = plt.subplots(figsize=(13,6))
sb.heatmap(data.corr(), annot=True, linewidth=.5, fmt='.1f', ax=ax)
plt.show()


# We see that there is no correlation between age and spending score of customer

# In[28]:


data.plot(kind='scatter', x='Age', y='Spending Score (1-100)', color='red', alpha=0.5, figsize=(7,7))
plt.xlabel('Age')
plt.ylabel('Spending score')
plt.show()


# In[27]:


data.Age.plot(kind='hist', figsize=(15,8), bins=50)
plt.xlabel('Age')
plt.show()


# We can see that in the graph, young customers are more than olders
