#!/usr/bin/env python
# coding: utf-8

# **This notebook is an exploratory analysis on Missing Migrant project dataset and is a work in progrss.
# **

# ![An image showing migrants crossing the Mediterrenian](https://theglobepost.com/wp-content/uploads/2018/08/African-migrants-boat.png)
# 
# As the war torn countries struggle and deaths toll rise, migrants crisis rises as one of the serious problem in the world. People have been fleeing their countries in any way possible looking for safer, better life for their family as refugees in the more developed countries. While this has been a hotly debated topic in current political scenario, one thing is undebatable; the hardship people go through to get out of their country with their lives hanging in the balance. Here we look at some of those unfortunate events and how bad the refugee crisis has been in recent times.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import folium
from folium.plugins import MarkerCluster
# Any results you write to the current directory are saved as output.


# **Reading in our csv file and parsing the column for dates**

# In[ ]:


dataframe = pd.read_csv('../input/MissingMigrants-Global-2019-03-29T18-36-07.csv', parse_dates=['Reported Date'])


# Here, we take a look into the data and try to get the sense of what we're working with and what we could like to know more about.

# In[ ]:


dataframe.head()


# In[ ]:


dataframe.dtypes


# **Graphing all the reported incidents**

# In[ ]:


dataframe.groupby('Reported Date')['Reported Date'].count().plot(kind='line', title="Incident Reports Graph", figsize=(14,6))


# **Taking a look at Number of incident report each year, as we can clearly see the trend is spiking every year with 2018 seeing more than 1400 incidents of Migrants disappearance / deaths**

# In[ ]:


dataframe.groupby('Reported Year')['Reported Year'].count().plot(kind='bar', title="Number of incidents based on year", figsize=(14,6))


# Since we are halfway through 2019, we compare the first six months data of every year to see if 2019 is following the same trend as the year before

# In[ ]:


monthstocheck=['Jan','Feb','Mar','Apr','May','Jun']
# data = dataframe[dataframe['Reported Month'].isin(monthstocheck)]['Reported Year'].value_counts()
# dataframe[dataframe['Reported Month'].isin(monthstocheck)]['Reported Year'].value_counts().sort_index(ascending=False).plot(kind='barh', title="Comparison First six months of 2019 with previous years", figsize=(10,6))
plt.figure(figsize=(10,6))
sns.barplot(dataframe[dataframe['Reported Month'].isin(monthstocheck)]['Reported Year'].value_counts().index,dataframe[dataframe['Reported Month'].isin(monthstocheck)]['Reported Year'].value_counts().values, alpha=0.8)
plt.ylabel("Number of Incidents")
plt.xlabel("Reported Year")
plt.title("Comparing first six months of 2019 with previous years")


# While it's a good news that 2019 has seen a lot less incidents than the previous years, the number is usually high in the later months for every year

# In[ ]:


dataframe.isna().sum()


# In[ ]:


dataframe.groupby('Reported Year')['Number Dead'].sum().astype(int).plot(kind='bar', title="Number of Deaths based on Year", figsize=(10,6))


# In[ ]:


dataframe.groupby('Reported Year')['Total Dead and Missing'].sum().astype(int).plot(kind='barh', title="Number of Dead/Missing per Year", figsize=(10,6))


# In[ ]:


df = dataframe.groupby('Reported Month').agg({'Number of Females':'sum','Number of Males':'sum'}).astype(int)
# x = dataframe['Reported Month'].value_counts().index.values
# y = dataframe['Reported Month'].value_counts().values
plt.figure(figsize=(12,6))
sns.lineplot(data=df)


# In[ ]:


dataframe.groupby('Reported Year').agg({'Number of Females':'sum','Number of Males':'sum'}).astype(int).plot(kind='bar',stacked=True, title="Number of Missing/Dead Female Vs Male", figsize=(10,6))


# In[ ]:


dataframe.head()


# Here we take a look at some of the most common cause of death. We make a word cloud at first followed by more illustrative bar graphs.

# In[ ]:


wordcloud = WordCloud(
    width = 1000,
    height = 600,
    background_color = 'white',
    stopwords = STOPWORDS).generate(str(dataframe['Cause of Death']))

fig = plt.figure(
    figsize = (20, 12))
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
# plt.tight_layout(pad=0)
plt.show()


# In[ ]:


dataframe['Cause of Death'].value_counts()[0:15].plot(kind='bar', title="Fifteen common causes of Death", figsize=(15,8))


# **Number of Incidents based on Region**

# In[ ]:


dataframe['Region of Incident'].value_counts().plot(kind='barh', title="Number of Incident report based on Region", figsize=(15,10))


# In[ ]:


dataframe.head()


# **Number of Deaths in every region**

# In[ ]:


dataframe.groupby('Region of Incident')['Total Dead and Missing'].sum().sort_values().plot(kind='bar', title="Dead/Missing Based of the Region", figsize=(12,8))


# In[ ]:


dataframe['Migration Route'].value_counts().sort_values().plot(kind='bar', title="Most common Routes for migrants", figsize=(14,6))


# Here we try to cluster all the incident report into an interactive map. You can see more detailed markers if you click on the clusters.

# In[ ]:


dataframe[['latitude','longitude']] = dataframe['Location Coordinates'].str.split(",",expand=True)


# In[ ]:


dataframe['latitude'] = dataframe['latitude'].astype(float).round(2)
dataframe['longitude'] = dataframe['longitude'].astype(float).round(2)


# In[ ]:


dataframe.head()


# In[ ]:


dataframe = dataframe.dropna(subset=['latitude','longitude'])


# In[ ]:


dataframe.isna().sum()


# In[ ]:


worldMap = folium.Map(zoom_start=16)


# In[ ]:


worldMap


# In[ ]:


mc = MarkerCluster()
for row in dataframe.itertuples():
    mc.add_child(folium.Marker(location=[row.latitude,row.longitude]))


# In[ ]:


worldMap.add_child(mc)

