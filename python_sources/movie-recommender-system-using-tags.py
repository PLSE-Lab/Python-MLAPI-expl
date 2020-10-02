#!/usr/bin/env python
# coding: utf-8

# # Movie Recommender
# 
# ### In this notebook, we try to build a movie recommender engine using the Tags of the movies(genome_scores.csv).

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


# ## Importing the dataset

# In[ ]:


scores=pd.read_csv('/kaggle/input/movielens-20m-dataset/genome_scores.csv')


# In[ ]:


movie=pd.read_csv('/kaggle/input/movielens-20m-dataset/movie.csv')


# #### Taking a Preview

# In[ ]:


scores.head()


# In[ ]:


scores.shape


# ### How many movies are there?

# In[ ]:


scores['movieId'].unique()


# In[ ]:


scores['movieId'].nunique()


# ### How many tags are associated with every movie?

# In[ ]:


scores['tagId'].unique()


# In[ ]:


scores['tagId'].nunique()


# In[ ]:


scores.describe()


# ## Making the pivot table with rows as `Movies`, columns as `Tags` and values as their `Relevance`

# In[ ]:


movie_tag_pivot=pd.pivot_table(columns='tagId',index='movieId',values='relevance',data=scores)


# In[ ]:


movie_tag_pivot


# #### Filling the missing values just in case there is any

# In[ ]:


movie_tag_pivot.fillna(-100,inplace=True)


# In[ ]:


from sklearn.neighbors import  NearestNeighbors


# In[ ]:


nn=NearestNeighbors(algorithm='brute',metric='minkowski',p=8)


# In[ ]:


nn.fit(movie_tag_pivot)


# ## Function to give the recommendations on providing a Movie Id 

# In[ ]:


def recommend(movie_id):
    distances,suggestions=nn.kneighbors(movie_tag_pivot.loc[movie_id,:].values.reshape(1,-1),n_neighbors=16)
    return movie_tag_pivot.iloc[suggestions[0]].index.to_list()


# ### Let's filter our movie DataFrame to have only the moovies that are there in the pivot_table

# In[ ]:


scores_movie=movie[movie['movieId'].isin(movie_tag_pivot.index.to_list())]


# ### Let's search for an Avengers movie

# In[ ]:


scores_movie[scores_movie['title'].str.contains('avengers',case=False)]


# ### Let's see what recommendations we get on passing Avengers(movie id: 89745) to our recommend function

# In[ ]:


recommendations=recommend(89745)


# In[ ]:


recommendations


# ### Let's see what these recommedations are!

# In[ ]:


for movie_id  in recommendations[1:]:
    print(movie[movie['movieId']==movie_id]['title'].values[0])


# ### Let's search for Harry Potter

# In[ ]:


scores_movie[scores_movie['title'].str.contains('harry potter',case=False)]


# ### Let's see what recommendations we get on passing Sorcerer's Stone(movie id: 4896) to our recommend function

# In[ ]:


recommendations=recommend(4896)


# In[ ]:


for movie_id  in recommendations[1:]:
    print(movie[movie['movieId']==movie_id]['title'].values[0])


# ### Let's dump our model

# In[ ]:


import pickle


# In[ ]:


pickle.dump(nn,open('engine.pkl','wb'))
pickle.dump(movie_tag_pivot,open('movie_tag_pivot_table.pkl','wb'))
pickle.dump(scores_movie,open('movie_names.pkl','wb'))


# In[ ]:




