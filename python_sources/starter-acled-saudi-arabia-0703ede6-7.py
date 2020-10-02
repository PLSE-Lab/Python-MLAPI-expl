#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# importing all required libraries

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


data = pd.read_csv('../input/acled_saudi.csv')


# In[ ]:


# inspecting columns 

data.columns 


# In[ ]:


# inspecting dataframe shape 

data.shape


# In[ ]:


# inspecting dataframe head 

data.head(3)


# In[ ]:


# checking null values
data.isnull().sum()


# In[ ]:


# exploring some features 

data.admin1.value_counts()


# In[ ]:


# exploring some features 

data.admin2.value_counts()[:15]


# In[ ]:


# checking data types

data.info()


# In[ ]:


# how many unique values in each column
data.nunique()


# In[ ]:


# fatalities distribution and outliers based on year
sns.boxplot(y='fatalities', data= data , x='year')
plt.show();


# In[ ]:


# fatalities distribution and outliers based on year and event type
sns.catplot(x='event_type' ,y='fatalities', data=data  , hue='year', height=6.5 , aspect=2.5 , kind='boxen')
plt.show();


# In[ ]:




