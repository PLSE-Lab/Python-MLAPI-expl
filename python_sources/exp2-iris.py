#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas
url="../input/iris.csv"
names=['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset=pandas.read_csv(url, names=names)


# In[2]:


#data slicing using .iloc
print(dataset.iloc[1:6, 4])


# In[3]:


#data slicing using .loc
print(dataset.loc[[5,20,70,41,120],['class']])


# In[4]:


#data slicing using .loc
dataset.loc[dataset['petal-length']==5.1,['class']]


# In[5]:


#data slicing using .ix
dataset.ix[[12,20,149,75,50],['sepal-length', 'sepal-width', 'petal-length', 'petal-width','class']]

