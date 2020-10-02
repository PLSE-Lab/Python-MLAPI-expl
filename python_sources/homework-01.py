#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv('../input/tmdb_5000_movies.csv')


# In[ ]:


data.info() # we get info about our dataframe


# In[ ]:


data.corr() # shows the correlation amongst the integer and float datas


# In[ ]:


# correlation map
f, ax = plt.subplots(figsize =(16,16))
sns.heatmap(data.corr(), annot = True, linewidths = 0.5, fmt = ".1f", ax=ax) 
# fmt .1 shows the digits that will be written after the dot
plt.show()


# In[ ]:


data.head(10) 
# shows the first 10 rows to get info about the big picture
# the default value shows 5 rows


# In[ ]:


data.columns 
# shows the column names


# In[ ]:


# Line Plot

data.budget.plot(kind = 'line', color = 'red', label = "Budget", linewidth = 1, alpha = .8, grid = True, linestyle = "-")
data.popularity.plot(kind = 'line', color = 'blue', label = "Popularity", linewidth = 1, alpha = .7, grid = True, linestyle = ":")

plt.legend(loc='upper right')
plt.xlabel('Budget')
plt.ylabel('Popularity')
plt.title('TMDB Movies Budget-Popularity')
plt.show()


# In[ ]:


# Scatter Plot

data.plot(kind = 'scatter', x='budget', y='popularity', alpha=0.3, color= "green")
plt.xlabel('Budget')
plt.ylabel('Popularity')
plt.title('TMDB Movies Budget-Popularity')


# In[ ]:


# Histogram

data.popularity.plot(kind='hist', bins=100,figsize=(16,16), color='purple')
plt.show()


# In[ ]:


# Dictionary

dict_01 = {'country_abv':'US', 'country_name' : 'United States of America'}
print(dict_01.keys())
print(dict_01.values())


# In[ ]:


dict_01['company']='Walt Disney' # adding new entry in dictionary
print(dict_01)


# In[ ]:


print('UK'in dict_01)     # checks whether 'UK' exists in dictionary


# In[ ]:


# Pandas

data = pd.read_csv('../input/tmdb_5000_movies.csv')


# In[ ]:


series = data['popularity']
print(type(series))
data_frame = data[['popularity']]
print(type(data_frame))


# In[ ]:


# Pandas_01 Filtering Pandas Data Frame

# To find out movies with popularity over 300 and budget under 150000000

x = data['popularity']>300
y = data['budget']<150000000
data[x&y]


# In[ ]:


# To find out Japanesse movies with an vote average over 8.0

a = data['vote_average']>8
b = data['original_language']=='ja'
data[a&b]


# In[ ]:





# In[ ]:


# Pandas_01 Filtering Pandas With logical_and

# To find out Japanesse movies with popularity over 100

data[np.logical_and(data['original_language']=='ja',data['popularity']>100)]


# In[ ]:


# This code gives the same output as the one in line 53

data[(data['original_language']=='ja') & (data['popularity']>100)]


# In[ ]:


# While and For Loops

dict_01 = {'country_abv':'US', 'country_name' : 'United States of America'}
for key,value in dict_01.items():
    print(key," : ",value)
print('')


# In[ ]:




