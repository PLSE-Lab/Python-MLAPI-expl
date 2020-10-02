#!/usr/bin/env python
# coding: utf-8

# # Movie Recommendation System

# Recommender systems are utilized in a variety of areas, and are most commonly recognized as playlist generators for video and music services like Netflix, YouTube and Spotify, product recommenders for services such as Amazon, or content recommenders for social media platforms such as Facebook and Twitter.
# <br>
# Recommender systems usually make use of either or both <b>Collaborative Filtering</b> or <b> Content-Based filtering approach. 

# ![](http://miro.medium.com/max/1600/1*dMR3xmufnmKiw4crlisQUA.png)

# Here we will be recommending movies to the users who have watched movie 'Jurassic Park (1993)' using <b> Collaborative Filtering</b>

# # Import necessary libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# # Import the data set

# Importing the rating data set which contains ratings given by the users to the movies they watched.

# In[ ]:


rating = pd.read_csv('../input/movielens-latest-small/ratings.csv')


# In[ ]:


rating.head()


# Importing the movie dataset which contains the description about all the movies

# In[ ]:


movies = pd.read_csv('../input/movielens-latest-small/movies.csv')


# In[ ]:


movies.head()


# Let's merge both the dataset so that in ratings dataset we have complete information about the movies apart from the movie id.

# In[ ]:


# merging both the datasets on 'movieId' column
movie_rating = pd.merge(left=rating,right=movies,on='movieId')


# In[ ]:


movie_rating.head()


# In[ ]:


movie_rating.columns


# Getting the columns of the movie_rating dataframe in proper order

# In[ ]:


movie_rating = movie_rating[['userId', 'movieId', 'title', 'genres', 'rating', 'timestamp']]


# In[ ]:


movie_rating.head()


# # Exploratory Data Analysis

# In[ ]:


movie_rating.info()


# In[ ]:


movie_rating.isnull().sum()


# Let's create a dataframe with number of ratings and average rating for each movie
# 

# In[ ]:


movie_rating.head(2)


# In[ ]:


# grouping the movies based on average rating
average_rating_movies = movie_rating.groupby('title')['rating'].mean().sort_values(ascending=False)


# In[ ]:


average_rating_movies.head(10)


# In[ ]:


average_rating_movies.hist(bins=20)
plt.show()


# Maximum movies have average rating in the range 3 to 4. The movies which have average = 5.0 may be the ones which may have been watched once or twice.

# In[ ]:


# grouping the movies based on count of users who rated the movies
count_userid = movie_rating.groupby('title')['userId'].count().sort_values(ascending=False)


# In[ ]:


count_userid


# In[ ]:


count_userid.hist()
plt.show()


# Maximum movies have been viewed in the range 0 - 40 views

# The movies which have average = 5.0 may be the ones which may have been watched once or twice. Let's see number of ratings given to movies which have average rating = 5.0

# In[ ]:


for movie in average_rating_movies[average_rating_movies==5.0].index:
    print(movie,count_userid[movie])


# In[ ]:


# grouping the movie_rating based on count on userId and mean on rating
userid_rating = movie_rating.groupby('title')[['userId','rating']].agg({'userId':'count','rating':'mean'}).round(2).sort_values(by='userId',ascending=False)


# In[ ]:


userid_rating.head()


# # Building Recommendation System

# In[ ]:


# creating pivot table to create item by item collaborative filtering
movie_rating_pivot = pd.pivot_table(index='userId',columns='title',values='rating',data=movie_rating)


# There will be many Nan values because users have watched only few of the movies and given ratings only to those movies

# In[ ]:


movie_rating_pivot.head()


# Most Rated movies:

# In[ ]:


userid_rating.head(10)


# Let's find which movies to recommend to the users who have watched 'Jurassic Park (1993)'. 
# To do this we have to find correlation of 'Jurassic Park (1993)' with other movies which have been rated in a similar way by the users.

# In[ ]:


# assigning ratings of movie 'Jurassic Park (1993)' to a new variable from movie_rating_pivot
jurassic_park = movie_rating_pivot['Jurassic Park (1993)'].head(10)


# In[ ]:


jurassic_park.head(10)


# Find the correlation with other movies from movie_rating_pivot table

# In[ ]:


correlation_jurassicpark = pd.DataFrame(movie_rating_pivot.corrwith(jurassic_park))


# In[ ]:


correlation_jurassicpark.head()


# Removing Nan values and naming the column as 'Correlation'

# In[ ]:


correlation_jurassicpark.columns = ['Correlation']
correlation_jurassicpark.dropna(inplace=True,axis=0)


# In[ ]:


correlation_jurassicpark.sort_values(by='Correlation',ascending=True).head()


# There may be movies which might have been watched only once or twice by the users who have watched 'Jurassic Park (1993)' and those movies will show high correlation. We will consider only those movies which have been viewed more than 100 times. Let's add views column in the correlation_jurassicpark data frame

# In[ ]:


correlation_jurassicpark['Views'] = userid_rating['userId']


# Now filtering out top 20 movies which have views greater than 100

# In[ ]:


correlation_jurassicpark[correlation_jurassicpark['Views'] > 100].sort_values(by='Correlation',ascending=False).head(20)


# In[ ]:




