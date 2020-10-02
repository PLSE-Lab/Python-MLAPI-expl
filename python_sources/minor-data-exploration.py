#!/usr/bin/env python
# coding: utf-8

# # **Preparing the environment and uploading data**

# **Import packages**

# In[ ]:


# data analysis and wrangling
import pandas as pd
import numpy as np

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(style='ticks')


# **Acquire data**

# In[ ]:


data = pd.read_csv("../input/weather/weather.csv", decimal=',')


# # **Exploratory Data Analysis (EDA)**

# In[ ]:


data.head()


# **Take detailed analysis of the data**
# <br>Let's create a function to simplify the analysis of general characteristics of the data. It contains info about types, counts, distincts, count nulls, missing ratio and uniques values of each feature.

# In[ ]:


def detailed_analysis(df): 
    obs = df.shape[0]
    types = df.dtypes
    counts = df.apply(lambda x: x.count())
    uniques = df.apply(lambda x: [x.unique()])
    nulls = df.apply(lambda x: x.isnull().sum())
    distincts = df.apply(lambda x: x.unique().shape[0])
    missing_ratio = (df.isnull().sum() / obs) * 100
    skewness = df.skew()
    kurtosis = df.kurt() 
    print('Data shape:', df.shape)
    
    cols = ['types', 'counts', 'distincts', 'nulls', 'missing ratio', 'uniques', 'skewness', 'kurtosis']
    details = pd.concat([types, counts, distincts, nulls, missing_ratio, uniques, skewness, kurtosis], axis = 1)
    
    details.columns = cols
    dtypes = details.types.value_counts()
    print('___________________________\nData types:\n',dtypes)
    print('___________________________')
    return details


# In[ ]:


details = detailed_analysis(data)
details


# **Explore the correlations**

# In[ ]:


correlations = data.corr()

fig = plt.figure(figsize=(12, 10))
sns.heatmap(correlations, annot=True, cmap='YlOrRd')


# **Relationships between data**

# In[ ]:


fig = plt.figure(figsize=(15, 10))
sns.regplot(x='temperature', y='humidity', data=data)


# In[ ]:


fig = plt.figure(figsize=(15, 10))
sns.boxplot(x='weather', y='temperature', hue='fire', data=data)


# In[ ]:


fig = plt.figure(figsize=(15, 10))
sns.barplot(x='weather', y='visibility', data=data, ci=None)


# In[ ]:


data[['weather', 'temperature']].groupby('weather').mean().sort_values(by='temperature', ascending=False)


# In[ ]:


data[['fire', 'wind']].groupby('fire').mean().sort_values(by='wind', ascending=False)

