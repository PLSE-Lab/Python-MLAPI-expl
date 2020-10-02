#!/usr/bin/env python
# coding: utf-8

# Many questions can be answered with this datasat, these are just a few:
# * Wildfire frequency per year 
# * Mean size of wildfire 

# In[7]:


import pandas as pd 
import matplotlib.pyplot as plt
import sqlite3


# In[8]:


# Connect to database
conn = sqlite3.connect('../input/FPA_FOD_20170508.sqlite')


# In[9]:


df = pd.read_sql_query("SELECT * From Fires", conn)
df.head()


# **What is the Wildfire Frequency Per Year?**

# In[10]:


# Create new DataFrame Populated with Relevant Data
df_year = df[['FIRE_YEAR']]


# In[11]:


df_year['FIRE_YEAR'].value_counts(sort=False).plot(kind="line",marker='o', figsize=(8,5))
plt.title('Frequency of Wildfires Over Time')
plt.xlabel('Year')
plt.ylabel('Number of Instances')
plt.show()


# **What is the Mean Size of a Wildfire?**

# In[12]:


df_size = df[['FIRE_SIZE']]
df_size = df_size.sort_values("FIRE_SIZE")

# Determine mean value
df_size["FIRE_SIZE"].mean()


# Looking at all instances, the mean size of a wildfire is approximately **75 acres**. 

# In[13]:


# Histogram plot of all instances 
df_size.plot(kind="hist")
plt.title('Distribution of Fire Sizes')
plt.xlabel('Fire Size(Acres)')
plt.ylabel('Number of Instances')
plt.show()


# We see that the above histogram isn't very useful. What if we take advantage of the 'FIRE_SIZE_CLASS'?

# In[14]:


# Bar plot with FIRE_SIZE_CLASS grouping
df_size_class = df[['FIRE_SIZE_CLASS']]
df_size_class.groupby('FIRE_SIZE_CLASS').size().plot(kind='bar', figsize=(8,5))
plt.title('Distribution of Fire Size Classes')
plt.xlabel('Fire Size Class')
plt.ylabel('Number of Instances')
plt.show()


# We see that the above graph is far more valuable, with most fires being classified as either 'A' or 'B', corresponding to fire sizes between 0 and 10 acres.

# 
