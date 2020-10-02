#!/usr/bin/env python
# coding: utf-8

# 1. Barplot using seaborn  
# 2. pandas's value_count()
# 3. Filtering using pandas

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from pandas import DataFrame , read_csv
#for f in os.listdir('../input/'):
#    print (f.ljust(30)+ str(round(os.path.getsize('../input/'+f)/10000, 2)) + 'MB')


# In[ ]:


df = pd.read_csv('../input/directory.csv')
df.head()


# In[ ]:


df.columns


# In[ ]:


# Plotting a bar graph of the number of stores in each city, for the first ten cities listed
# in the column 'City'
city_count  = df['City'].value_counts()
city_count = city_count[:10,]
plt.figure(figsize=(10,5))
sns.barplot(city_count.index, city_count.values, alpha=0.8)
plt.title('Starbucks in top 10 cities in the World')
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('city', fontsize=12)
plt.show()


# In[ ]:


# Plotting a bar graph of the number of stores in each city, for the first ten cities listed
# in the column 'City'
city_count  = df['Country'].value_counts()
city_count = city_count[:10,]
plt.figure(figsize=(10,5))
sns.barplot(x=city_count.index, y=city_count.values, alpha=0.8)
plt.title('Starbucks in top 10 countries in the World')
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('city', fontsize=12)
plt.show()


# In[ ]:





# In[ ]:


newyorkTimezone = df[df.Timezone == 'GMT-05:00 America/New_York']['City'].value_counts()[:10]
plt.figure()
chart=sns.barplot(newyorkTimezone.index, newyorkTimezone.values, alpha=0.8)
plt.title('Starbucks in top 10 cities in New yord timezone')
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('city', fontsize=15)
chart.set_xticklabels(chart.get_xticklabels(), rotation=45)
plt.show()
# chart = sns.barplot(x = 'genre', y = 'popularity', hue = 'mode', data = df.sample(50))


# plt.title('Popularity Based on Mode and Key')


# In[ ]:


plt.figure()
continents = df.Timezone.apply(lambda s: s.split('/')[0].split()[-1]).value_counts()
sns.barplot(continents.index, continents.values)
plt.title('# of Starbucks stores by Continent')
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Continent', fontsize=12)
plt.show()


# In[ ]:


df.Timezone.str.contains('Etc').mask(lambda c: c == False).dropna().index


# In[ ]:


df[df.Timezone.str.contains('Etc')]


# In[ ]:


df.iloc[df.Timezone.str.contains('Etc').where(lambda x: x == True).dropna().index]


# In[ ]:


df.Timezone.str.contains('Etc').filter(items = [11314])


# In[ ]:




