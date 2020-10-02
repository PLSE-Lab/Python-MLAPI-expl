#!/usr/bin/env python
# coding: utf-8

# **Wikipedia Page View - Data Analysis**

# In[ ]:


#importing libaries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from datetime import datetime


# In[ ]:


#loading dataset
data = pd.read_csv('../input/wikipediapageviewschristmas/pageviews-20180226-20200226.csv',header='infer')


# **Data Exploration**

# In[ ]:


data.shape


# In[ ]:


#Checking for null/missing values
data.isna().sum()


# In[ ]:


#Checking for column data types
data.dtypes


# In[ ]:


data.head()


# In[ ]:


plt.figure(figsize=(15,6))
plt.title('Distribution of Views per day', fontsize=16)
plt.tick_params(labelsize=14)
sns.distplot(data['Christmas'], bins=60);


# In[ ]:


#Converting date to Pandas DateTime
data['Converted_Date'] = pd.to_datetime(data['Date'])


# In[ ]:


# add column 'Day', 'Month', 'Year' to the dataframe
data['Day'] = data['Converted_Date'].dt.day
data['Month'] = data['Converted_Date'].dt.month
data['Year'] = data['Converted_Date'].dt.year


# In[ ]:


#Converting the date column to index
data.index = pd.DatetimeIndex(data['Date'])
data = data.drop(columns=['Date','Converted_Date'],axis=1)


# In[ ]:


data.head()


# In[ ]:


view_pivot = pd.pivot_table(data, values='Christmas', index=['Month'],
                    columns=['Year'])

view_pivot.plot(figsize=(10,8))


# In[ ]:


sns.set(rc={'figure.figsize':(11, 4)})
data['Christmas'].plot(linewidth=0.5);


# In[ ]:


ax = data.loc['2019':'2020', 'Christmas'].plot(marker='o', linestyle='-')
ax.set_ylabel('Views');

