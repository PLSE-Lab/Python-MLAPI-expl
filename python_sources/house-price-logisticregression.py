#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


# In[13]:


data = pd.read_csv('../input/train.csv')
data.head()


# In[14]:


data_null = data.isnull().sum().sort_values(ascending=False)
data_null[data_null>0]


# In[15]:


data.fillna(0,inplace=True)
data.head()


# In[16]:


sns.distplot(data['SalePrice'])


# In[18]:


categoricals= []
for col,col_type in data.dtypes.iteritems():
    if col_type =='O':
        categoricals.append(col)
    else:
        data[col].fillna(0,inplace=True)


# In[19]:


data=pd.get_dummies(data,columns=categoricals,dummy_na=True)
data.head()


# In[20]:


dependent_variable= "SalePrice"
x= data[data.columns.difference([dependent_variable])]
y = data[dependent_variable]
lr = LogisticRegression()
lr.fit(x,y)


# In[21]:


lr.score(x,y)

