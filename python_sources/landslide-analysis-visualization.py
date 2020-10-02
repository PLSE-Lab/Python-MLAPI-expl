#!/usr/bin/env python
# coding: utf-8

# ![Landslide_pic][1]
# [1]: http://smalworld.co/wp-content/uploads/2017/12/new-140108-utah-avalanche-jms-1851_6dc27b33b29e6d7ba2efadf13dcfdf90.nbcnews-fp-1200-800-1024x683.jpg
# # The purpose of this kernel:
# * Learning to clean data for use.
# * Learning basic visualization tools (Seaborn methods e.g Bar, Point, Joint, Count, Pie, Lm, Kde, Box, Swarm, Pair Plots)
# * Relationships between landslide and population of countries. 

# ## Update
# * I'll make updates on this kernel when I learned new things about data visualization. You can check regularly to see new information.

# ## Contents
# 1. [Loading Data and First Look:](#1)
#     1. [Importing Modules](#2)
#     1. [Reading Data](#3)
#     1. [Checking the data for missing values](#4)
# 1. [Cleaning Data](#5)
# 1. [Data Analysis](#6)
#     1. [Population of American Countries Affected by Landslides](#7)
#     1. [20 Most Affected States/Provinces by Landslides](#8)
#     1. [Landslide Occurrences  in United States, Colombia and Mexico 2007 - 2016](#9)
# 1. [Summary](#20)
#     
#     

# <a id="1"></a> <br>
# # Loading Data and First Look
# 

# <a id="2"></a> <br>
# ### Importing Modules

# In[ ]:


import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import os
print(os.listdir("../input"))


# <a id="3"></a> <br>
# ### Reading Data

# In[ ]:


data = pd.read_csv('../input/catalog.csv')


# <a id="4"></a> <br>
# ### Checking the data for missing values

# In[ ]:


data.columns


# In[ ]:


data['id'].value_counts(dropna=False)


# In[ ]:


data['date'].value_counts(dropna=False)


# In[ ]:


data['time'].value_counts(dropna=False)


# In[ ]:


data['continent_code'].value_counts(dropna=False)


# In[ ]:


data['country_name'].value_counts(dropna=False)


# In[ ]:


data['country_code'].value_counts(dropna=False)


# In[ ]:


data['state/province'].value_counts(dropna=False)


# In[ ]:


data['population'].value_counts(dropna=False)


# In[ ]:


data['city/town'].value_counts(dropna=False)


# In[ ]:


data['distance'].value_counts(dropna=False)


# In[ ]:


data['location_description'].value_counts(dropna=False)


# In[ ]:


data['latitude'].value_counts(dropna=False)


# In[ ]:


data['longitude'].value_counts(dropna=False)


# In[ ]:


data['geolocation'].value_counts(dropna=False)


# In[ ]:


data['hazard_type'].value_counts(dropna=False)


# In[ ]:


data['landslide_type'].value_counts(dropna=False)


# In[ ]:


data['landslide_size'].value_counts(dropna=False)


# In[ ]:


data['trigger'].value_counts(dropna=False)


# In[ ]:


data['storm_name'].value_counts(dropna=False)


# In[ ]:


data['injuries'].value_counts(dropna=False)


# In[ ]:


data['fatalities'].value_counts(dropna=False)


# In[ ]:


data['source_name'].value_counts(dropna=False)


# In[ ]:


data['source_link'].value_counts(dropna=False)


# <a id="5"></a> <br>
# # Cleaning Data
# * **ID**'s are not in order. There're 1692 entries for this dataset but ID's start from 34 end with 7541. We should fix this.
# * There are NaN values in **time** column. Also we have 'evening' and 'Evening' which are the same. We should name them 'Evening'.
# * There are missing values for **continent_code**. Only South America (SA) is avaible. It's better to drop this column since I won't be using it.
# * There are missing values for **location_description** and unknown/other values. We can classify these values to 'Unknown' as well.
# * There some errors in **landslide_type**. We have 'Landslide', 'Mudslide' and 'landslide', 'mudslide' which are the same. We should rename them.
#     Also there are NaN, Other and Unknown values. We should classify them as Other.
# * There are some errors in **landslide_size**. We have 'Large', 'Medium', 'Small' and 'large', 'medium', 'small' which are the same. We should rename them.
#     Also there is one NaN value we might need to handle.
# * There are four trigger in **trigger** column that needs to be handled. 'Unknown', 'unknown', 'NaN', 'Other'. We can classify  these values to 'Unknown'
# * There is one lowercase downpour in the data. We can make it uppercase.
# * There are missing values for **source name** and **source_link**. We can classify these values to 'Unknown'.

# ### Deleting the contitent_code column

# In[ ]:


data.drop(['continent_code'], axis=1, inplace=True)


# ### Gathering all unknown, other, NaN values to single value 'Unknown'

# In[ ]:


data.trigger = ['Unknown' if i=='unknown' else
                'Unknown' if i=='Other' else
                'Downpour' if i=='downpour' else
                i
                for i in data.trigger]
# We can also use replace method
# data.trigger.replace(['unknown'],'Unknown', inplace=True)
# data.trigger.replace(['Other'],'Unknown', inplace=True)
# data.trigger.replace(['downpour'],'Downpour', inplace=True)

data.trigger = data.trigger.fillna('Unknown')

data.location_description = ['Unknown' if i=='Other' else
                             i
                             for i in data.location_description
                            ]
data.location_description = data.location_description.fillna('Unknown')

data.time = ['Evening' if i=='evening' else
              i
              for i in data.time]
data.time = data.time.fillna('Unknown')

data.landslide_type = ['Landslide' if i=='landslide' else
                       'Mudslide' if i=='mudslide' else
                       'Other' if i=='Unknown' else 
                        i 
                        for i in data.landslide_type]
data.landslide_type = data.landslide_type.fillna('Other')

data.landslide_size = ['Large' if i=='large' else
                       'Medium' if i=='medium' else
                       'Small' if i=='small' else
                        i
                        for i in data.landslide_size]

data.source_name = data.source_name.fillna('Unknown')
data.source_link = data.source_link.fillna('Unknown')

# There is no need in this dataset but for example; if the population values were object (string) type values we can't use this data in numerical operations. So we need to convert it into integer values.
# To do this:
# data.population = data.population.astype(int)  # other types are str, float


# ### Checking the types of the values

# In[ ]:


data.info()
data.head()


# ### Date entries are string values. We can convert them to datetime64[ns].

# In[ ]:


data.date = pd.to_datetime(data.date)


# ### Fixing the ID's

# In[ ]:


data.id = range(0,1693)


# <a id="6"></a> <br>
# # Data Analysis

# <a id="7"></a> <br>
# ### Population of American Countries Affected by Landslides

# In[ ]:


# We need these columns to calculate the affected population
data1= data[['country_name','state/province', 'city/town', 'population']]
data1


# In[ ]:


# Landslides can occur more than once in the same city/town.
# We need to remove duplicates or we'll be making mistake when calculating polulation.
data1 = data1.drop_duplicates(subset=['city/town'])  # removing duplicates
data1


# In[ ]:


# Calculating the population for the same country
data1['total_population'] = data1.groupby(['country_name'])['population'].transform('sum')
data1


# In[ ]:


# We've calculated the total population of each country. We can delete the population column.
data1.drop(['population'], axis=1, inplace=True)  # removing population values
data1


# In[ ]:


# We have duplicates for the same country. We need to remove them.
data1 = data1.drop_duplicates(subset=['country_name'])
data1


# In[ ]:


# Also we don't need state/province and city/town values anymore. We can delete them.
data1.drop(['state/province', 'city/town'], axis=1, inplace=True)
data1


# In[ ]:


# We can sort our values by total population in order to see descending graphic
data1.sort_values(by='total_population', ascending=False, inplace=True)
data1


# ### Seaborn Barplot

# In[ ]:


plt.figure(figsize=(15,10))  # Size of the grapgic
sns.barplot(x=data1['country_name'], y=data1['total_population'])  # Defining x and y axis
plt.xticks(rotation=90)  # to show xlabels(country names) 90 degrees (vertical text)
plt.xlabel('Country')
plt.ylabel('Affected Population')
plt.title('Population of American Countries Affected by Landslides 2007-2016')
plt.show()


# <a id="8"></a> <br>
# ### 20 Most Affected States/Provinces by Landslides

# In[ ]:


# We need to calculate how many state/province are there in our data
data2= data[['country_name','state/province']]
data2


# In[ ]:


data2Group = data2.groupby('state/province').count()  # counting state/provinces
data2Group = data2Group.rename(columns={'country_name': 'count'})  
# After counting, these values will be written in the country_name column. In order the reduce the confusion we can rename our column to 'count'
data2Group


# In[ ]:


# We need most 20 state/province so we should sort our list
data2Group.sort_values(by='count', ascending=False, inplace=True)  # sorting our new data
data2Group


# In[ ]:


# Selecting most 20 state/provice and writing them in a new dataframe called 'most20state'
most20state = data2Group.head(20)
most20state = most20state.reset_index()  # to move out state/province values to column
most20state


# ### Seaborn Barplot (cubehelix_palette)

# In[ ]:


plt.figure(figsize=(15,10))
sns.barplot(x=most20state['state/province'], y=most20state['count'], palette=sns.cubehelix_palette(20))
plt.xlabel('States / Provinces')
plt.ylabel('Number of Landslides')
plt.xticks(rotation=45)
plt.title('20 Most Affected States / Provinces by Landslides 2007 - 2016')
plt.show()


# <a id="9"></a> <br>
# ### Landslide Occurrences  in United States, Colombia and Mexico 2007 - 2016
# 

# In[ ]:


# Creating a new dataframe for plotting
data3 = data[['date', 'country_name']]
data3


# In[ ]:


# Filtering the data
filterUSA = data['country_name'] == 'United States'
filterMex = data['country_name'] == 'Mexico'
filterCol = data['country_name'] == 'Colombia'
dataUSA = data3[filterUSA]
dataMex = data3[filterMex]
dataCol = data3[filterCol]
dataUSA


# In[ ]:


# To create time series data we need to set the data index to date
dataUSA = dataUSA.set_index('date')
dataMex = dataMex.set_index('date')
dataCol = dataCol.set_index('date')
dataMex


# In[ ]:


# Resampling the data according to years('A') and count the frequency of landslides
dataUSA = dataUSA.resample('A').count()
dataMex = dataMex.resample('A').count()
dataCol = dataCol.resample('A').count()
dataCol


# In[ ]:


# Moving the data index('date') to a column
dataUSA = dataUSA.reset_index()
dataMex = dataMex.reset_index()
dataCol = dataCol.reset_index()
dataUSA


# In[ ]:


# We don't need month and day values in the 'date' column. We can extract the year with this code:
dataUSA.date = [i.year for i in dataUSA.date]
dataMex.date = [i.year for i in dataMex.date]
dataCol.date = [i.year for i in dataCol.date]
dataMex


# In[ ]:


# Renaming the column name to reduce confusion.
dataUSA = dataUSA.rename(columns={'country_name': 'count'})
dataMex = dataMex.rename(columns={'country_name': 'count'})
dataCol = dataCol.rename(columns={'country_name': 'count'}) 
dataCol


# ## Seaborn Pointplot

# In[ ]:


f, ax = plt.subplots(figsize=(15,10))
sns.pointplot(x='date', y='count', data=dataUSA, color='blue')
sns.pointplot(x='date', y='count', data=dataMex, color='red')
sns.pointplot(x='date', y='count', data=dataCol, color='green')
plt.text(0,60,'United States', color='blue', fontsize=15, style='italic')  # 0-> position of x value, 60-> positiob of y value
plt.text(0,12,'Mexico', color='red', fontsize=15, style='italic')
plt.text(0,-7,'Colombia', color='green', fontsize=15, style='italic')
plt.xlabel('Year', fontsize=20)
plt.ylabel('Landslide Occurrences', fontsize=20)
plt.xticks(rotation=45)
plt.title('Landslide Occurrences in United States, Colombia and Mexico 2007 - 2016', fontsize=20, style='oblique')
plt.grid()
plt.show()


# <a id="20"></a> <br>
# # Summary
# * Learned how to clean a data and make preparations for plotting.
# * Learned how to plot data by using Seaborn Barplot and Pointplot.
# 
