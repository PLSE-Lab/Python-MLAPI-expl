#!/usr/bin/env python
# coding: utf-8

# ## Importing Libraries and Data set

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Importing Data set
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data = pd.read_csv('/kaggle/input/netflix-shows/netflix_titles.csv')
dataCopy = data
data.head(10)


# ## Checking out data types and missing data

# In[ ]:


#Checking columns names
list(data.columns)


# In[ ]:


data.info()


# In[ ]:


data.shape[0] # number of entries


# We have some missing values in Director, Country, Cast and Date added. We will deal with it when we visualize it

# #  Number of Movies vs Number of Series

# In[ ]:


sns.set(style='darkgrid')
sns.countplot(x = 'type',data=data,palette='Set1')


# We see that Netflix has way more number of Movies than TV Shows

# # Which country creates most number of Shows/Movies in Netflix?

# In[ ]:


plt.figure(figsize=(12,10))
plt.title('Top 15 Countries on the basis of Content Creation')
sns.countplot(data=data,y='country',order=data['country'].value_counts().index[0:15],palette='Accent')


# US Produces most number of shows!

# ## In which year highest number of Shows were added?

# In[ ]:


plt.figure(figsize=(12,10))
sns.set(style='darkgrid')
ax = sns.countplot(y='release_year',data=data,order=data['release_year'].value_counts().index[0:15],palette='Set1')


# In 2018 Most number of shows and movies were added!

# <H1> MOVIES </H1>

# In[ ]:


movie = data[data['type']=='Movie']
#movie.columns
movie.head(5)


# <H2>What is the average duration of Movies in Netflix?</H2>

# In[ ]:


duration = []
movie = movie[movie['duration'].notna()]
for i in movie['duration']:
    duration.append(int(i.strip('min')))


# In[ ]:


plt.figure(1,figsize=(15,10))
plt.title("Duration of Movies")
sns.distplot(duration)


# Most number of Movies run for 90 minutes.

# ## Director with Most movies

# In[ ]:


plt.figure(figsize=(15,8))
plt.title('Directors with Most movies')
sns.countplot(y='director',data=movie,order=movie['director'].value_counts().index[0:10],palette='Set3')


# Raul Campos has directed Most Number of Movies

# ## Most movies from a Specific Genre

# In[ ]:


genrePerMovie=[]
totalMoveGenre = []
setGenre = set()
set1 = set()
for i in movie['listed_in']:
    if(type(i)==str):
        g = i.split(',')
        for genre in g:
            setGenre.add(genre.strip())


# In[ ]:


totalMovieGenre = list(setGenre)
#len(totalMovieGenre)


# In[ ]:


get_ipython().run_cell_magic('time', '', "storeCountOfGenre = {}\ncurrentGenre = []\nfor actualGenre in totalMovieGenre:\n    count = 1\n    for i in movie['listed_in']:\n        currentGenre = []\n        if(type(i)==str):\n            s=i.split(',')\n            for j in s:\n                currentGenre.append(j.strip())\n            if(actualGenre in currentGenre):\n                if actualGenre not in storeCountOfGenre:\n                    storeCountOfGenre[actualGenre] = 1\n                else:\n                    storeCountOfGenre[actualGenre] +=1")


# In[ ]:


import operator
import itertools

sorted_Genre = dict(sorted(storeCountOfGenre.items(), key=operator.itemgetter(1),reverse=True))
finalSortedListOfGenre = dict(itertools.islice(sorted_Genre.items(),10))

keysGenre = list(finalSortedListOfGenre.keys())
keysGenre = keysGenre[1:]
valuesGenre = list(finalSortedListOfGenre.values())
valuesGenre = valuesGenre[1:]

#[1:] is done because after all this calculations, 'Internal Movies' came up in the top as most movie had this Genre.
#But 'International Movie' is not a genre. So a temporary solution is to remove 1st element from value and key list


# In[ ]:


import matplotlib.cm as cm
from matplotlib.colors import Normalize
from numpy.random import rand
dataColorGenre = [2, 3, 5, 6, 8, 12, 7, 5,9,11,10,4]
my_cmap = cm.get_cmap('ocean')
my_norm = Normalize(vmin=0, vmax=8)

plt.figure(figsize=(10,7))
plt.bar(keysGenre, valuesGenre, color=my_cmap(my_norm(dataColorGenre)))
plt.xticks(rotation=90)
plt.show()


# Dramas tops the list followed by Comedy Movies and then Documentries

# ## Casts with most number of Movies in Netflix

# In[ ]:


castPerMovie=[]
totalCast = []
set1 = set()
for i in movie['cast']:
    if(type(i)==str):
        s = i.split(',')
        for j in s:
            set1.add(j.strip())


# In[ ]:


# Run this to check if any cast has repeated ( CROSS VERIFY IF THE ABOVE CODE WORKS AS EXPECTED)
#from collections import Counter
#Counter(l)


# In[ ]:


totalCast = list(set1)
#len(totalCast)


# In[ ]:


get_ipython().run_cell_magic('time', '', "storeCounts = {}\ncurrentCasts = []\nfor actualCast in totalCast:\n    count = 1\n    for i in movie['cast']:\n        currentCasts = []\n        if(type(i)==str):\n            s=i.split(',')\n            for j in s:\n                currentCasts.append(j.strip())\n            if(actualCast in currentCasts):\n                if actualCast not in storeCounts:\n                    storeCounts[actualCast] = 1\n                else:\n                    storeCounts[actualCast] +=1")


# In[ ]:


sorted_d = dict(sorted(storeCounts.items(), key=operator.itemgetter(1),reverse=True))
finalSortedList = dict(itertools.islice(sorted_d.items(),20))

keys = finalSortedList.keys()
values = finalSortedList.values()


# In[ ]:


plt.figure(figsize=(10,5))

dataColorDirector = [2, 3, 5, 6, 8, 12, 7, 5,9,11,10,4]
my_cmap = cm.get_cmap('rainbow')
my_norm = Normalize(vmin=0, vmax=8)

plt.bar(keys,values,color=my_cmap(my_norm(dataColorDirector)))
plt.xticks(rotation=90)


# Anupam Kher tops the list with most number of movies

# ## TV Series

# In[ ]:


data = dataCopy
series = data[data['type']=='TV Show']
series.head(5)


# ## Average No. of Seasons in TV Shows

# In[ ]:


durationSeries = []
tvshow = series[series['duration'].notna()]
for i in tvshow['duration']:
    durationSeries.append(int(i.strip('Season')))

plt.figure(figsize=(12,10))
plt.title('Average no. of Seasons of TV Shows')
sns.distplot(durationSeries)


# Most of the TV Shows in Netflix have 1 Season

# ## Country with most no. of available TV Shows

# In[ ]:


setCountry = set()
for country in series['country']:
    if(type(country) == str):
        s = country.split(',')
        for singleCountry in s:
            setCountry.add(singleCountry.strip())


# In[ ]:


#setCountry
totalCountriesForSeries = list(setCountry)
#len(totalCountriesForSeries)


# In[ ]:


get_ipython().run_cell_magic('time', '', "currentCountries = []\ncountCountryTvShows = dict()\nfor singleCountry in totalCountriesForSeries:\n    currentCountries = []\n    for country in series['country']:\n        if(type(country)==str):\n            s = country.split(',')\n            for j in s:\n                currentCountries.append(j.strip())\n            if(singleCountry in currentCountries):\n                if(singleCountry not in countCountryTvShows):\n                    countCountryTvShows[singleCountry] = 1\n                else:\n                    countCountryTvShows[singleCountry]+=1")


# In[ ]:


sorted_countCountriesTvShows = dict(sorted(countCountryTvShows.items(), key=operator.itemgetter(1),reverse=True))
finalSortedDictTvShows = dict(itertools.islice(sorted_countCountriesTvShows.items(),10))

keys = finalSortedDictTvShows.keys()
values = finalSortedDictTvShows.values()


# In[ ]:


#finalSortedDictTvShows

plt.figure(figsize=(10,5))

dataColorDirector = [1, 2, 3, 4, 5, 6, 7, 8,9,10,11,12]
my_cmap = cm.get_cmap('jet')
my_norm = Normalize(vmin=0, vmax=8)

plt.bar(keys,values,color=my_cmap(my_norm(dataColorDirector)))
plt.xticks(rotation=90)


# All these countries have 1600+ tv shows available in their Country

# ## Cast with Most number of TV Shows

# In[ ]:


get_ipython().run_cell_magic('time', '', "castPerShow=[]\ntotalCastTvShow = []\ntvShowset = set()\nfor i in series['cast']:\n    if(type(i)==str):\n        s = i.split(',')\n        for j in s:\n            tvShowset.add(j.strip())\n            \ntotalCastTvShow = list(tvShowset)\nlen(totalCastTvShow)")


# In[ ]:


get_ipython().run_cell_magic('time', '', "storeCountCastTvShow = {}\ncurrentCastsTvShow = []\nfor actualTvShowCast in totalCastTvShow:\n    for i in series['cast']:\n        currentCastsTvShow = []\n        if(type(i)==str):\n            s=i.split(',')\n            for j in s:\n                currentCastsTvShow.append(j.strip())\n            if(actualTvShowCast in currentCastsTvShow):\n                if actualTvShowCast not in storeCountCastTvShow:\n                    storeCountCastTvShow[actualTvShowCast] = 1\n                else:\n                    storeCountCastTvShow[actualTvShowCast] +=1\n                    \nsortedCastTvShow = dict(sorted(storeCountCastTvShow.items(), key=operator.itemgetter(1),reverse=True))\nfinalSortedListTvShowCast = dict(itertools.islice(sortedCastTvShow.items(),10))\n\nkeys = finalSortedListTvShowCast.keys()\nvalues = finalSortedListTvShowCast.values()")


# In[ ]:


plt.figure(figsize=(10,5))

dataColorDirector = [12, 11, 10, 9, 8, 7, 6, 5,4,3,2,1]
my_cmap = cm.get_cmap('flag')
my_norm = Normalize(vmin=0, vmax=8)

plt.bar(keys,values,color=my_cmap(my_norm(dataColorDirector)))
plt.title("Top 10 Casts with no. of Shows")
plt.xticks(rotation=90)


# Takahiro Sakurai has most number of shows in Netflix

# ## Genre with most number of TV Shows

# In[ ]:


genrePerShow=[]
totalGenreShow = []
setGenreShow = set()
setShowGenre = set()
for i in series['listed_in']:
    if(type(i)==str):
        g = i.split(',')
        for genre in g:
            setShowGenre.add(genre.strip())
            
totalShowGenre = list(setShowGenre)
len(totalShowGenre)


# In[ ]:


get_ipython().run_cell_magic('time', '', "storeCountOfShowGenre = {}\ncurrentShowGenre = []\nfor actualShowGenre in totalShowGenre:\n    for i in series['listed_in']:\n        currentShowGenre = []\n        if(type(i)==str):\n            s=i.split(',')\n            for j in s:\n                currentShowGenre.append(j.strip())\n            if(actualShowGenre in currentShowGenre):\n                if actualShowGenre not in storeCountOfShowGenre:\n                    storeCountOfShowGenre[actualShowGenre] = 1\n                else:\n                    storeCountOfShowGenre[actualShowGenre] +=1")


# In[ ]:


sortedShowGenre = dict(sorted(storeCountOfShowGenre.items(), key=operator.itemgetter(1),reverse=True))
finalSortedListOfShowGenre = dict(itertools.islice(sortedShowGenre.items(),11))

keysShowGenre = list(finalSortedListOfShowGenre.keys())
keysShowGenre = keysShowGenre[1:]

valuesShowGenre = list(finalSortedListOfShowGenre.values())
valuesShowGenre = valuesShowGenre[1:]

#print(keysShowGenre,valuesShowGenre)
#[1:] is done because after all this calculations, 'Internal Movies' came up in the top as most movie had this Genre.
#But 'International Movie' is not a genre. So a temporary solution is to remove 1st element from value and key list


# In[ ]:


dataColorGenre = [2, 3, 5, 6, 8, 12, 7, 5,9,11,10,4]
my_cmap = cm.get_cmap('ocean')
my_norm = Normalize(vmin=0, vmax=8)

plt.figure(figsize=(10,7))
plt.bar(keysShowGenre, valuesShowGenre, color=my_cmap(my_norm(dataColorGenre)))
plt.title("Top 10 Genre in TV Shows")
plt.xticks(rotation=90)
plt.show()


# Shows with genre TV Dramas are appx 600 followed by Comedies and Crime Tv Shows

# In[ ]:




