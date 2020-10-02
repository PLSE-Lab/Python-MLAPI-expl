#!/usr/bin/env python
# coding: utf-8

# **INTRO:**
# 
# **This is my first data analysis.
# Please feel free to share your feedback & knowledge so that I can improve**
# 
# Thanks,
# Aditya Totla

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #for plotting
import seaborn as sea #for visualization

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


#importing datasets 

movies = pd.read_csv("../input//imdb-20002019-movie-data-set-over-3000/df_movie_ratings.csv")

#show the columns

movies.columns


# In[ ]:


movies.head()


# In[ ]:


#Removing Unwanted Column
movies = movies.drop(movies.columns[0], axis='columns')


# In[ ]:


movies.columns


# In[ ]:


#we have 3 float columns, 3 integer and 3 object columns according to info() method

movies.info()


# In[ ]:


#some numeric informations about the movies_df

movies.describe()


# ***Lets Do Some Visualization to Get the Insights from the Data***

# In[ ]:


#Correlation map with using seaborn lib.

movies_corr = movies.corr()
f,ax = plt.subplots(figsize=(10, 10))
sea.heatmap(movies_corr, annot = True, linewidths = 0.1, fmt= '.2f', ax=ax )
plt.show()


# In[ ]:


# these are the rating point in the database

print("Rating Points :",movies['imdb'].unique())


# In[ ]:


# lets see how many films are there for each rating point

print(movies['imdb'].value_counts())


# In[ ]:


# Visualizing rating points using pie chart

plt.figure(1, figsize=(10,10))
movies['imdb'].value_counts().plot.pie(autopct="%1.1f%%")


# In[ ]:


#scatter plot of movie and their ratings between 2000 - 2020

plt.scatter(movies.year, movies.imdb, alpha = 0.07, label = "Movie", color = "orange")
plt.xlabel("Years")
plt.ylabel("Ratings")
plt.legend(loc = "lower right")
plt.show()


# In[ ]:


#histogram plot about number of published movies according to year

movies.year.plot(kind = "hist", bins = 40, figsize = (12,8))
plt.xlabel("Years")
plt.ylabel("Number of Movies")
plt.show()


# In[ ]:


movies["runtime"].value_counts()


# In[ ]:


movies.runtime.value_counts()[:10].plot.pie(autopct='%1.1f%%',figsize=(10,10))
plt.title('Top 10 runtime of Movies')


# In[ ]:


movies['runtime'] = movies['runtime'].map(lambda x: x.lstrip(' ').rstrip('min'))


# In[ ]:


movies_time=movies.runtime
f,ax = plt.subplots(figsize=(14, 8))
sea.distplot(movies_time, bins=20, kde=False,rug=True, ax=ax);
plt.ylabel("Counts")


# **Let's see if critics and users get along really well.**

# In[ ]:


movies['imdb'].corr(movies['metascore'])


# **There seems to be a correlation, let's visualize.**

# In[ ]:


import matplotlib
matplotlib.style.use('ggplot')

plt.scatter(movies.metascore, movies.imdb)
plt.show()

