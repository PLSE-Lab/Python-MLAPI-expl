#!/usr/bin/env python
# coding: utf-8

# #### Things I have analysed here are
# * Which streaming platform(s) can I find this movie on?
# * Average IMDb ratings of a movie produced in a country?
# * Target age group movies vs the streaming application they can be found on
# * The year during which a movie was produced and the streaming platform they can be found on
# * Analysis of the popularity of a movie vs year
# 

# **Data visualisation of this data set**
# [https://public.tableau.com/profile/arul.rajan.v#!/vizhome/movies_OTT_platform/Story1](http://)

# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Lets see the visualitation of the movies avaiable in the OTT platform and Lets see which platform is best for subscription
# 
# 

# In[ ]:


from matplotlib import pyplot as plt
import seaborn as sns


# In[ ]:


movie = pd.read_csv('/kaggle/input/movies-on-netflix-prime-video-hulu-and-disney/MoviesOnStreamingPlatforms_updated.csv')
movie


# In[ ]:


movie.head()


# In[ ]:


movie.info()


# Dropping the rotten tomatoes ratings very few movies rating are available

# In[ ]:


movie.drop("Rotten Tomatoes",axis=1,inplace=True)


# 

# In[ ]:


#check the null count
movie.isnull().sum()


# In[ ]:


print('Total number of movies                : ',movie['Title'].count() )
print('Total number of movies in Netflix     : ',movie['Netflix'].sum() )
print('Total number of movies in Hulu        : ',movie['Hulu'].sum() )
print('Total number of movies in Prime Video : ',movie['Prime Video'].sum() )
print('Total number of movies in Disney+     : ',movie['Disney+'].sum() )


# From the above data, we can find that Amazon prime has lots of movies

# In[ ]:


data = [movie['Netflix'].sum(),movie['Hulu'].sum(),movie['Prime Video'].sum(),movie['Disney+'].sum()]
my_labels = 'Netflix','Hulu','Prime','Disney+'
plt.pie(data, labels=my_labels, autopct='%1.1f%%', startangle=15)
plt.title('Movies in different platforms')
plt.axis('equal')
plt.show()


# In[ ]:


set(movie.Country.values)


# In[ ]:



len(set(movie.Country.values))


# In[ ]:


movie.max()


# In[ ]:


plt.figure(figsize = (10, 10))
movie.plot(x='Year',y='Runtime',kind='scatter',color='R')
plt.show()


# In[ ]:


seperated_genres = movie['Genres'].str.get_dummies(',')
seperated_genres


# In[ ]:


seperated_genres.sum()


# In[ ]:


plt.figure(figsize = (10, 10))
seperated_genres.sum().plot(kind="bar")
plt.ylabel('Genres')
plt.xlabel('Total number of movies')
plt.title('Movies and its genres')
plt.show()


# In[ ]:


movie.columns


# In[ ]:


len(set(movie.Language.values))


# In[ ]:


movie.Language.unique()


# In[ ]:



top_30_screenplay = movie.sort_values(by = 'Runtime', ascending = False).head(30)
plt.figure(figsize = (15, 10))
sns.barplot(data = top_30_screenplay, y = 'Title', x = 'Runtime', hue = 'Country', dodge = False)
plt.legend(loc = 'lower right')
plt.xlabel('Total minutes')
plt.ylabel('Movie')
plt.title('Top 30 movies by Run Time')
plt.show()


# In[ ]:


print(movie.groupby("Age").count()["ID"])

sns.countplot(data=movie,x="Age")


# In[ ]:


n = 50

top_ratings = movie.sort_values(by="IMDb",ascending=False).reset_index().iloc[:n]
top_ratings


# In[ ]:


new_movies = movie[movie['Year'] > 2010]
new_movies


# In[ ]:


print('Total number of new movies                : ',new_movies['Title'].count() )
print('Total number of new movies in Netflix     : ',new_movies['Netflix'].sum() )
print('Total number of new movies in Hulu        : ',new_movies['Hulu'].sum() )
print('Total number of new movies in Prime Video : ',new_movies['Prime Video'].sum() )
print('Total number of new movies in Disney+     : ',new_movies['Disney+'].sum() )


# In[ ]:


plt.figure(figsize = (10, 15))
new_movies.plot(x='Year',y='Runtime',kind='scatter',color='R')
plt.show()


# In[ ]:


x = new_movies["Year"]
y = new_movies["IMDb"]

plt.figure(figsize=(8,8))
sns.scatterplot(x,y,data=new_movies,)


# In[ ]:


rate_mov = movie[movie['IMDb'] > 8]


# In[ ]:


print("Total Movies with more than 8+ rating(IMDb)     : ", rate_mov['ID'].count())
print('Total number of movies 8+ rating in Netflix     : ',rate_mov['Netflix'].sum() )
print('Total number of movies 8+ rating in Hulu        : ',rate_mov['Hulu'].sum() )
print('Total number of movies 8+ rating in Prime Video : ',rate_mov['Prime Video'].sum() )
print('Total number of movies 8+ rating in Disney+     : ',rate_mov['Disney+'].sum() )


# In[ ]:


top_rated_data = pd.DataFrame({
    'platforms' : ['Netflix', 
                   'Disney', 
                   'Prime Video', 
                   'Hulu'],
    'total_mov' : [rate_mov['Netflix'].sum(),
                   rate_mov['Disney+'].sum(),
                   rate_mov['Prime Video'].sum(),
                   rate_mov['Hulu'].sum()]
})


# In[ ]:


plt.figure(figsize = (10, 10))
sns.barplot(data = top_rated_data,
           x = top_rated_data['platforms'],
           y = top_rated_data['total_mov']
)
plt.ylabel('Platform')
plt.xlabel('Total number of 8+ rated movies')
plt.title('Platform with most movies rated above 8+ (IMDB)')
plt.show()


# In[ ]:


india = movie[movie.Country =='India']
india


# In[ ]:


print('Total number of Indian movies                : ',india['Title'].count() )
print('Total number of Indian movies in Netflix     : ',india['Netflix'].sum() )
print('Total number of Indian movies in Hulu        : ',india['Hulu'].sum() )
print('Total number of Indian movies in Prime Video : ',india['Prime Video'].sum() )
print('Total number of Indian movies in Disney+     : ',india['Disney+'].sum() )


# In[ ]:


indian_mov = india[india['IMDb'] > 7]
print("Total Movies with more than 7+ rating(IMDb)     : ", indian_mov['ID'].count())
print('Total number of movies 8+ rating in Netflix     : ', indian_mov['Netflix'].sum() )
print('Total number of movies 8+ rating in Hulu        : ',indian_mov['Hulu'].sum() )
print('Total number of movies 8+ rating in Prime Video : ',indian_mov['Prime Video'].sum() )
print('Total number of movies 8+ rating in Disney+     : ',indian_mov['Disney+'].sum() )


# In[ ]:





# In[ ]:


tamil = movie[movie.Language =='Tamil']
tamil


# In[ ]:


print('Total number of Tamil movies                : ',tamil['Title'].count() )
print('Total number of Tamil movies in Netflix     : ',tamil['Netflix'].sum() )
print('Total number of Tamil movies in Hulu        : ',tamil['Hulu'].sum() )
print('Total number of Tamil movies in Prime Video : ',tamil['Prime Video'].sum() )
print('Total number of Tamil movies in Disney+     : ',tamil['Disney+'].sum() )


# In[ ]:


# Total tamil movies and the years realeased and IMDB rating
x = tamil["Year"]
y = tamil["IMDb"]

plt.figure(figsize=(10,10))
m = sns.scatterplot(x,y,data=tamil,)


# In[ ]:





# On analysis, 
# * we can find that there are 16744 number of movies available in all OTT platforms.
#     * Total number of movies in Netflix     :  3560
#     * Total number of movies in Hulu        :  903
#     * Total number of movies in Prime Video :  12354
#     * Total number of movies in Disney+     :  564

# ** Top 3 genres**
# * Comedy         4637
# * Drama          7227
# * Thriller       3354
# 

# **Movies released after 2010 available in OTT are**
# * Total number of new movies                :  9231
# * Total number of new movies in Netflix     :  2884
# * Total number of new movies in Hulu        :  676
# * Total number of new movies in Prime Video :  5930
# * Total number of new movies in Disney+     :  155

# * Total Movies with more than 8+ rating(IMDb)     :  478
# * Total number of movies 8+ rating in Netflix     :  129
# * Total number of movies 8+ rating in Hulu        :  23
# * Total number of movies 8+ rating in Prime Video :  324
# * Total number of movies 8+ rating in Disney+     :  21
# 
# 
# Check out my tableau vizzes:
# Data visualisation of this data set https://public.tableau.com/profile/arul.rajan.v#!/vizhome/movies_OTT_platform/Story1

# In[ ]:




