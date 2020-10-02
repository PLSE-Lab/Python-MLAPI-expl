#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


data = pd.read_csv(r'/kaggle/input/nigerian-market-fires-2020/Market Fires 2020.csv')
data.head()


# In[ ]:


data.dtypes


# In[ ]:


#lets convert the Date of Fire to Datetime.
data['Date of Fire'] = pd.to_datetime(data['Date of Fire'])


# In[ ]:


# Lets check the value count for unique state
pd.value_counts(data['State']).plot.bar()


# In[ ]:


pd.value_counts(data['Reported Causes']).plot.bar()


# In[ ]:


pd.value_counts(data['Fire put out by']).plot.bar()


# In[ ]:


pd.value_counts(data['Type of Market']).plot.bar()


# In[ ]:


data['Month'] = data['Date of Fire'].dt.month
data.head()


# In[ ]:


pd.value_counts(data['Month']).plot.pie()


# In[ ]:


#Please comment, Thank you.
#Thank you for the dataset

