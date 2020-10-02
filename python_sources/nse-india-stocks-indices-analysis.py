#!/usr/bin/env python
# coding: utf-8

# # importing the liabraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import squarify
get_ipython().run_line_magic('matplotlib', 'inline')


# # reading the datasets

# In[2]:


df1=pd.read_csv('../input/banknifty.csv')
df2=pd.read_csv('../input/nifty50.csv')


# In[3]:


df1.head()


# In[4]:


df2.head()

# aggregating the information about Bank Nifty Dataset
# In[5]:


df1.info()


# # aggregating the information about the Nifty50 Dataset

# In[6]:


df2.info()


# transforming the date columns to a generalized format

# In[7]:


df1['date']=pd.to_datetime(df1['date'].astype(str), format='%Y%m%d')


# In[8]:


df2['date']=pd.to_datetime(df2['date'].astype(str), format='%Y%m%d')


# Deleting the irrelevant columns so as to reduce the complexity of the data

# In[9]:


del df1['index']


# In[10]:


del df2['index']


# grouping the data with relevance to date

# In[11]:


date_df1=df1.groupby('date').mean()


# In[ ]:


date_df2=df2.groupby('date').mean()


# In[12]:


date_df1.head()


# In[13]:


date_df1.describe()


# In[15]:


date_df2.head()


# # Visualization of the Bank Nifty Data w.r.t Date factor

# In[16]:


plt.figure(figsize=(20,7))
plt.plot(date_df1,color='blue')
plt.title('Bank Nifty Graph')


# # Visualization of the Nifty50 Data w.r.t Date factor

# In[17]:


plt.figure(figsize=(20,7))
plt.plot(date_df2,color='blue')
plt.title('Nifty50 Graph')


# # Visualization using KDE graph

# In[18]:


sns.kdeplot(data=df1['high'],gridsize=50)


# In[22]:


sns.kdeplot(data=df2['high'],gridsize=50)


# # Analysis of correlation using Heat Maps

# In[23]:


sns.heatmap(df1.corr(),annot=True)


# In[24]:


sns.heatmap(df2.corr(),annot=True)


# In[ ]:




