#!/usr/bin/env python
# coding: utf-8

# # Introduction
# Recommender Systems:
# 1. Rating-user Based Recommender Systems
# 2. Tag-user Based Recommender Systems
# 
# **The purpose of recommender systems is recommending new things that are not seen before from people.**
# 
# We will use Collaborative Filtering while recommending
# 
# **Collaborative filtering means to recommend according to the combination of your experience and experiences of other people.**

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# Importing the necessary data-set

# In[ ]:


#importing the genome_scores dataset to get the tag scores
scores=pd.read_csv('../input/genome_scores.csv')
scores.columns


# In[ ]:


#importing the link dataset to get the tag scores
link=pd.read_csv('../input/link.csv')
link.columns


# In[ ]:


# we need all the columns
scores.head()


# In[ ]:


# import movie data set and look at columns
movie = pd.read_csv("../input/movie.csv")
movie.columns


# In[ ]:


# what we need is that movie id and title
movie = movie.loc[:,["movieId","title"]]
movie.head(10)


# In[ ]:


#importing rating data set
rating = pd.read_csv("../input/rating.csv")
rating.columns


# In[ ]:


# what we need is that user id, movie id and rating
rating = rating.loc[:,["userId","movieId","rating"]]
rating.head(10)


# *1.Starting with* **rating-user recommendation system.**

# In[ ]:


# then merge movie and rating data
data = pd.merge(movie,rating,on='movieId')


# In[ ]:


# now lets look at our data 
data.head()


# In[ ]:


data.shape


# We see a huge amount data. All of these data will take a lot of time to work with. So we will consider only those data which will help us to recommend and will not take un-neccessary space.
# 
# Here we have a lot of user-id. We will consider only those users that has rated for more than 300 movies.
# 
# **Reason -** A person with good movie knowledge has definitely seen and rated more movies, and we will consider only those users.
# 
# 
# *This process will automatically reduce my data without making problem to my analysis and prediction.*

# In[ ]:


#getting the no. of times each user rated

a=data['userId'].value_counts().reset_index()
a.rename(columns={'userId':'count','index':'userId'},inplace = True)
a


# In[ ]:


a.shape


# In[ ]:


#we will consider only those users who have rated for more than 300 times

a = a[a['count']>300]
a.shape


# In[ ]:


#we will consider only those selected users only in our analysis

data = data[data['userId'].isin(a['userId'])]
data.shape


# Our data is finally reduced.
# 
# Now we will go on with our predictions.

# In[ ]:


# lets make a pivot table in order to recommend easily
ptable = data.pivot_table(index = ["movieId"],columns = ["userId"],values = "rating")
ptable.head()


# **Will feel all the NaN values with 0**

# In[ ]:


ptable=ptable.fillna(0)


# In[ ]:


ptable.head()


# * As it can be seen from table above, rows are movie Id, columns are user Id and values are ratings

# **Getting inside my algorithm**

# In[ ]:


#importing the necessary package

from sklearn.neighbors import NearestNeighbors


# In[ ]:


model=NearestNeighbors(algorithm='brute')


# In[ ]:


model.fit(ptable)


# Function to recommend movies

# In[ ]:


def recommends(movie_id):
    distances,suggestions=model.kneighbors(ptable.loc[movie_id,:].values.reshape(1,-1),n_neighbors=16)
    return ptable.iloc[suggestions[0]].index


# In[ ]:


l=movie[movie['movieId'].isin(ptable.index)]


# Checking for the movie 'Avengers'

# In[ ]:


l[l['title'].str.contains('avengers',case=False)]


# getting the movie code and will recommend using the code itself

# In[ ]:


recommendation=recommends(89745)


# In[ ]:


#getting the recommend movie's Id

recommendation


# In[ ]:


#getting the movie names from it's Id

for movie_id  in recommendation[1:]:
    print(movie[movie['movieId']==movie_id]['title'].values[0])


# We will check for Harry Potter now

# In[ ]:


l[l['title'].str.contains('harry potter',case=False)]


# In[ ]:


recommendation=recommends(4896)

#getting the movie names from it's Id

for movie_id  in recommendation[1:]:
    print(movie[movie['movieId']==movie_id]['title'].values[0])


# 2.Starting with tag-user recommendation system

# Doing the same procedures in this case again.

# In[ ]:


scores.head()


# In[ ]:


scores.shape


# In[ ]:


movie_tag_pivot=pd.pivot_table(columns='tagId',index='movieId',values='relevance',data=scores)


# In[ ]:


movie_tag_pivot


# In[ ]:


movie_tag_pivot.fillna(0,inplace=True)


# In[ ]:


model1=NearestNeighbors(algorithm='brute')


# In[ ]:


model1.fit(movie_tag_pivot)


# In[ ]:


def recommend(movie_id):
    distances,suggestions=model1.kneighbors(movie_tag_pivot.loc[movie_id,:].values.reshape(1,-1),n_neighbors=16)
    return movie_tag_pivot.iloc[suggestions[0]].index


# In[ ]:


#we will merge the link and scores dataset now

movie = pd.merge(movie,link,on='movieId')


# In[ ]:


scores_movie=movie[movie['movieId'].isin(movie_tag_pivot.index)]


# Predictions for the movie avengers

# In[ ]:


scores_movie[scores_movie['title'].str.contains('avengers',case=False)]


# In[ ]:


recommendations=recommend(89745)


# In[ ]:


recommendations


# In[ ]:


for movie_id  in recommendations[1:]:
    print(movie[movie['movieId']==movie_id]['title'].values[0])


# In[ ]:


scores_movie[scores_movie['title'].str.contains('harry potter',case=False)]


# In[ ]:


recommendation=recommend(4896)

for movie_id  in recommendation[1:]:
    print(movie[movie['movieId']==movie_id]['title'].values[0])


# # We see both the models are predicting more or less the same type of things.
# 
# (Anyway from an user perspective, our recommendation is working good.)

# Model Dumping

# In[ ]:


import pickle as pkl


# In[ ]:


#for tag-user recommendation

pkl.dump(model1,open('engine_tu.pkl','wb'))
pkl.dump(movie_tag_pivot,open('movie_tag_pivot_table_tu.pkl','wb'))
pkl.dump(scores_movie,open('movie_names_tu.pkl','wb'))


# In[ ]:


#one problem will persist while dumping the rating vs user pivot table. that is we have seen that the data is huge in that table and we might face space problem in this IDE.

#thus we will consider the dump file of tag-user only.


# In[ ]:




