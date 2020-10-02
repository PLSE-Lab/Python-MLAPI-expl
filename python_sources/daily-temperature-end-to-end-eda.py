#!/usr/bin/env python
# coding: utf-8

# **Import Required libraries**

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.core.display import HTML

import warnings


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')
warnings.filterwarnings('ignore')


# Check directories where data is

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Load Data to Pandas DataFrame

# In[ ]:


df=pd.read_csv('/kaggle/input/daily-temperature-of-major-cities/city_temperature.csv')


# # Inspect Data

# Check size of the data

# In[ ]:


df.shape


# Get the number of rows/records only

# In[ ]:


df.shape[0]


# Get the number of columns only

# In[ ]:


df.shape[1]


# Show column names

# In[ ]:


df.columns


# Show first 5 rows/records

# In[ ]:


df.head()


# Show last 5 rows/records

# In[ ]:


df.tail()


# Check data types

# In[ ]:


df.dtypes


# Convert Data Types for Day, Month & Year to String

# In[ ]:


df['Day_STR']=df['Day'].astype(str)
df['Month_STR']=df['Month'].astype(str)
df['Year_STR']=df['Year'].astype(str)


# In[ ]:


df.dtypes


# Show more information including memory

# In[ ]:


df.info()


# check for missing data

# In[ ]:


df.isna().sum()


# Check for Outliers using boxplot

# In[ ]:


fig, ax = plt.subplots(figsize=(16,8))
sns.boxplot(x='Region',y='AvgTemperature',data=df,ax=ax)


# Remove records with -99.0 as average Temperature

# In[ ]:


# Check the number of records
df[df['AvgTemperature']==-99.0].count()


# In[ ]:


# Remove these records
df=df.drop(df[df['AvgTemperature']==-99.0].index)


# In[ ]:


df.head()


# Add Date Column

# In[ ]:


df['Date']=df['Day'].astype(str)+'/'+df['Month'].astype(str)+'/'+df['Year'].astype(str)


# In[ ]:


df.head()


# Convert created date to date formart

# In[ ]:


df['Date']=pd.to_datetime(df['Date'])


# In[ ]:


df.dtypes


# In[ ]:


df.head()


# Recheck out data size

# In[ ]:


df.shape


# # Explore Data

# How many unique Countries do we have in our data

# In[ ]:


df['Country'].nunique()


# How many unique Coutries per Region are we having

# In[ ]:


df.groupby(['Region'])['Country'].nunique()


# In[ ]:


df.head()


# Get average temperature per Region throughout the years

# In[ ]:


df.groupby(['Region'])['AvgTemperature'].mean()


# Pivot Region & Year data with average temperature for flourish racing bar chart

# In[ ]:


pivoted_df=pd.pivot_table(df[['Region','AvgTemperature','Year']], 
                          values='AvgTemperature', index=['Region'],
                          columns=['Year'], aggfunc=np.mean)


# In[ ]:


pivoted_df.head()


# In[ ]:


pivoted_df.to_csv('region_temperature_racing_bar_chart.csv')


# Daily Average Temperature Per Region

# In[ ]:


HTML('''<div class="flourish-embed flourish-bar-chart-race" data-src="visualisation/2738038" data-url="https://flo.uri.sh/visualisation/2738038/embed"><script src="https://public.flourish.studio/resources/embed.js"></script></div>''')


# In[ ]:


df.groupby(['Region'])['AvgTemperature'].mean().plot(kind='bar',figsize=(17,7))


# Show top 10 Countries with maximum temperature Per Region

# In[ ]:


df.groupby(['Region','Country'])['AvgTemperature'].max().sort_values(ascending=False).head(10)


# Show top 10 Countries with minimum temperature

# In[ ]:


df.groupby(['Country'])['AvgTemperature'].min().sort_values(ascending=False).head(10)


# Pivot Country & Year data with average temperature for flourish racing bar chart

# In[ ]:


pivoted_Country_Year_df=pd.pivot_table(df[['Country','AvgTemperature','Year']], 
                          values='AvgTemperature', index=['Country'],
                          columns=['Year'], aggfunc=np.mean)


# In[ ]:


pivoted_Country_Year_df.head()


# In[ ]:


pivoted_Country_Year_df.to_csv('country_temperature_racing_bar_chart.csv')


# Average Daily Temperature Per Country Per Year

# In[ ]:


HTML('''<div class="flourish-embed flourish-bar-chart-race" data-src="visualisation/2738445" 
data-url="https://flo.uri.sh/visualisation/2738445/embed">
<script src="https://public.flourish.studio/resources/embed.js"></script></div>''')


# Show minimum temperature in Kenya

# In[ ]:


df[df['Country']=='Kenya'].groupby('Country')['AvgTemperature'].min()


# Show maximum temperature in Kenya

# In[ ]:


df[df['Country']=='Kenya'].groupby('Country')['AvgTemperature'].max()


# Show the average AvgTemperature temperature in Kenya

# In[ ]:


df[df['Country']=='Kenya'].groupby('Country')['AvgTemperature'].mean()


# Show the top 5 Cities with the highest temperature in the dataset

# In[ ]:


df.groupby('City')['AvgTemperature'].max().sort_values(ascending=False).head(10)


# In[ ]:


df.groupby('City')['AvgTemperature'].max().sort_values(ascending=False).head(30).plot(kind='bar',figsize=(17,7))


# Show top 5 hottest days in Asia

# In[ ]:


df[df['Region']=='Asia'].groupby(['Country','Year','Month','Day'])['AvgTemperature'].max().sort_values(ascending=False).head()


# Show temperature trend on a year & Month basis

# In[ ]:


df.groupby(df['Date'].dt.to_period('M'))['AvgTemperature'].mean().plot(kind='line',figsize=(17,7))


# # Statistical Functions and Analysis

# Show Highest temperature

# In[ ]:


df['AvgTemperature'].max()


# Show lowest temperature

# In[ ]:


df['AvgTemperature'].min()


# Show Average tempetarure

# In[ ]:


df['AvgTemperature'].mean()


# Show standard deviation for the AvgTemperature

# In[ ]:


df['AvgTemperature'].std()


# Show Variance in AvgTemperature

# In[ ]:


df['AvgTemperature'].var()


# Show Skew

# In[ ]:


df['AvgTemperature'].skew()


# Show Kurtosis

# In[ ]:


df['AvgTemperature'].kurt()


# In[ ]:


sns.distplot(df['AvgTemperature'])


# Show Statistical Summary

# In[ ]:


df.describe(include='all')


# Line plot

# In[ ]:


sns.relplot(x='Year',y='AvgTemperature',data=df,kind='line',hue='Region',height=16, aspect=1)


# Distribution plots

# In[ ]:


sns.pairplot(data=df[['AvgTemperature','Region','Month']],hue='Region',height=7, aspect=1)

