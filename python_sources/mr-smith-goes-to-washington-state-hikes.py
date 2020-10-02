#!/usr/bin/env python
# coding: utf-8

# ---PROJECT SCOPE---
# Exploratory Analysis: 
# - Create five scatterplots, plotting summit success (%) against each metric for tracking weather. 1) temperature 2) radiation 3) humidity 4) wind direction 5) windspeed. 
# - Each scatterplot will segment each data point based on the route of the summit attempt (Disappointment Cleaver, Emmons-Winthrop, Other) based on color.
# Deeper Analysis and Final Deliverable TBD based on observations from above scatterplots.
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import missingno as mno


# In[ ]:


# Any results you write to the current directory are saved as output.
weather_filepath = "../input/mount-rainier-weather-and-climbing-data/Rainier_Weather.csv"
weather_data = pd.read_csv(weather_filepath, index_col=0, encoding="latin-1")
weather_data.head()


# In[ ]:


climbing_filepath = "../input/mount-rainier-weather-and-climbing-data/climbing_statistics.csv"
climbing_data = pd.read_csv(climbing_filepath, index_col=0, encoding="latin-1")
climbing_data.head()


# First, we need to perform... a JOIN on the two tables to create a new one. 

# In[ ]:


result = pd.merge(weather_data, 
                  climbing_data[['Route','Success Percentage']],
                  on='Date',
                 how='left')
result.head()


# The merge seems to have worked. However, the problem is that climbing data goes back farther than weather data, by a whopping nine months:

# In[ ]:


result.tail()


# I don't want to try and find an external data source to fill in these values, so I'll remove all the dates that don't have both climbing and weather data. I'm thinking the best way to do this would simply be to change the type of MERGE to "inner", which should only leave a subset of the data where both tables have values for a specific date.

# In[ ]:


result = pd.merge(weather_data, 
                  climbing_data[['Route','Success Percentage']],
                  on='Date',
                 how='inner')
result.tail()


# Looks alright. If I weren't documenting the process, I would have taken out the last several lines of code and markdown, but what we have now should be a full subset of the earlier data set. 
# 
# Let's attempt a scatterplot:

# In[ ]:


TempGraph = result.filter(['Temperature AVG', 'Success Percentage'], axis=1)
TempGraph.tail()


# In[ ]:


sns.scatterplot(x="Success Percentage", y="Temperature AVG", data=TempGraph)


# Five of the values here don't make any sense. How can there be more than a 100% chance of summiting on a given day?
# 
# I looked at a few different options for removing outliers on Stack Overflow. First I'll try defining the 99th percentile of the success data, and then filter everything below the 99th percentile:

# In[ ]:


quantile = TempGraph["Success Percentage"].quantile(0.99)
TempGraph = TempGraph[TempGraph["Success Percentage"] < quantile]


# In[ ]:


sns.jointplot(x='Success Percentage', y='Temperature AVG', data=TempGraph, kind='reg')


# I have two problems with this visualization. For one, it doesn't provide a correlation coefficient. More importantly, if you don't look at the mini visualization above the range of the y-axis, it's hard to realize just how many data points lie at 0%. Let's try a hexplot instead:

# In[ ]:


sns.jointplot(x='Success Percentage', y='Temperature AVG', data=TempGraph, kind='hex')


# Summiting looks bleak no matter how balmy the weather. If we wanted to inspire false confidence in our intrepid hikers, we could exlude days that don't have a single successful summit by adding "xlim=(0.1,1)" to the hexplot's parameters. 
# 
# Now, as for separating each data point into one of three categories based on climbing route:

# In[ ]:


result = result[result['Success Percentage'] < result['Success Percentage'].quantile(0.99)]
sns.lmplot(x='Success Percentage', y='Temperature AVG', hue = 'Route', data=result)


# Well isn't this a disaster! Let's filter out all but the two most common routes, Disappointment Cleaver and Emmons-Winthrop. By renaming all the categories in question to the same name, we can place them all into the same bucket. 
# 
# 

# In[ ]:


result['Route'] = np.where((result['Route'] != "Disappointment Cleaver"), "Other", result['Route'])
sns.lmplot(x='Success Percentage', y='Temperature AVG', hue = 'Route', data=result)


# In[ ]:


result.corr()

Solar radiation and temperature seem to be the only strong correlation here. And with that, I'm going to end this notebook. I don't see anything particularly interesting to visualize, and I'm not interested in working with prediction just yet.