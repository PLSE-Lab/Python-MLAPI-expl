#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import plotly.graph_objs as go
import plotly.figure_factory as ff
import plotly.express as px


# Reading and preprocessing data as pandas dataframes

# In[ ]:


users = pd.read_csv('/kaggle/input/movietweetings/users.dat', sep='::', header=None)
users.columns = ['user_id', 'twitter_id']
ratings = pd.read_csv('/kaggle/input/movietweetings/ratings.dat', sep='::', header=None)
ratings.columns = ['user_id', 'movie_id', 'rating', 'rating_timestamp']
movies = pd.read_csv('/kaggle/input/movietweetings/movies.dat', sep='::', header=None)
movies.columns = ['movie_id', 'movie_title', 'genres']


# Merging datasets into one dataframe

# In[ ]:


df = pd.merge(users, ratings, on='user_id', how='outer')
df = pd.merge(df, movies, on='movie_id', how='outer')
df = df.sort_values('rating_timestamp')
df


# One hot encoding for 'genres' column

# In[ ]:


df['genres'] = df['genres'].str.split('|')
genres = df['genres'].str.join('|').str.get_dummies()
genres = genres.reset_index()
genres = genres.drop(['index'],axis=1)
genres


# In[ ]:


df = df.reset_index()
df = df.drop(['index', 'genres'], axis=1)
df


# In[ ]:


df = pd.concat([df, genres], axis=1)
df


# In[ ]:


df = df.drop(['user_id', 'movie_id'],axis=1)
df


# In[ ]:


fig = px.histogram(df, "rating", nbins=11, title='Rank distribution')
fig.show()


# In[ ]:


data = df.groupby(['movie_title'])['rating'].mean().reset_index().sort_values('rating').tail(20)
fig = px.bar(data, x="rating", y="movie_title", orientation='h', title='Top 20 movies with the highest ranking')
fig.show()


# In[ ]:


tweets = df['movie_title'].value_counts().reset_index()
tweets.columns = ['movie_title', 'total_tweets']
df = pd.merge(df, tweets, on='movie_title', how='inner')
df


# In[ ]:


data = df[df['total_tweets']>100].groupby(['movie_title'])['rating'].mean().reset_index().sort_values('rating').tail(20)
fig = px.bar(data, x="rating", y="movie_title", orientation='h', title='Top 20 movies with the at least 100 votes')
fig.show()


# In[ ]:




