#!/usr/bin/env python
# coding: utf-8

# ### Hi everyone! In this kernel I will show a simple recommendation system based on this data.
# 
# First of all, I must say that there are many types of recommendation systems, but usually there are 2 types:
# - Content Based
# - Collaborative filtering
# 
# More information here - www.towardsdatascience.com/what-are-product-recommendation-engines-and-the-various-versions-of-them-9dcab4ee26d5

# In[ ]:


import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

from surprise import Reader, Dataset, SVD


# In[ ]:


movies = pd.read_csv('../input/movielens-dataset/movies.csv')


# In[ ]:


movies.head()


# #### Let's start with content based recommender system.
# The essence of such a system is that we look for similar films by similar characteristics. Only genres are known about films. Let's try to write a system based only on this.

# In[ ]:


movies['genres'] = movies['genres'].apply(lambda x: x.replace('|', ' '))


# In[ ]:


movies.head()


# We will use the CountVectorizer to convert a collection of text documents into a token matrix.

# In[ ]:


count = CountVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
count_matrix = count.fit_transform(movies['genres'])


# In[ ]:


count_matrix.shape


# There are many ways to determine the similarity, but I will use the cosine similarity.

# In[ ]:


cosine_sim = cosine_similarity(count_matrix, count_matrix)


# In[ ]:


cosine_sim.shape


# In[ ]:


movies = movies.reset_index()
titles = movies['title']
indices = pd.Series(movies.index, index=movies['title'])


# Write a simple function to get top similar movies.

# In[ ]:


def get_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    movie_indices = [i[0] for i in sim_scores]
    return titles.iloc[movie_indices]


# We can notice that even such a simple system gives good results.

# In[ ]:


get_recommendations('Toy Story (1995)').head(5)


# #### Now write a recommendation system for collaborative filtering.
# The peculiarity of such a system is that we are now looking for not similar films, but similar users. We assume that if we can find users who like the same films, we can recommend films that one of them has not seen, but rated well.

# In[ ]:


ratings = pd.read_csv('../input/movielens-dataset/ratings.csv')


# In[ ]:


ratings.head()


# I will use one of my favorite libraries to create recommendation systems - surprise.

# In[ ]:


reader = Reader()


# In[ ]:


data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)


# We will use matrix decompositions. They help a lot in creating recommendation systems.

# In[ ]:


svd = SVD()


# In[ ]:


trainset = data.build_full_trainset()
svd.fit(trainset)


# In[ ]:


svd.predict(1, 101).est


# We predict that user number 1 will rate movie number 101 as 3.54

# In[ ]:




