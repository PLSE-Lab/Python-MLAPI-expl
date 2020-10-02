#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
df = pd.read_csv("../input/autompg-dataset/auto-mpg.csv", na_values='?')
df.head(5)


# In[ ]:


df.describe()


# In[ ]:


df.columns


# In[ ]:


df.isnull().any()


# In[ ]:


horsepower_median = df['horsepower'].median()


# In[ ]:


df['horsepower'].fillna(horsepower_median,inplace=True)


# In[ ]:


df['horsepower'].isnull().any()


# In[ ]:


df.boxplot(column = [df.columns[0],df.columns[1]])


# In[ ]:


df.boxplot(column = [df.columns[2],df.columns[3]])


# In[ ]:


df.boxplot(column = [df.columns[4]])


# In[ ]:


df.boxplot(column = [df.columns[5]])


# In[ ]:


df.boxplot(column = [df.columns[6]])


# In[ ]:


df.boxplot(column = [df.columns[7]])


# **So from this we can see that the columns acceleration and horsepower has many outlier values**

# In[ ]:


#Do outlier detection
            

