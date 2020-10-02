#!/usr/bin/env python
# coding: utf-8

# # **The Movie Recommender Systems**

# Let's load the data now.

# In[ ]:


import pandas as pd 
import numpy as np 


# In[ ]:


from surprise import Reader, Dataset, SVD, evaluate
reader = Reader()
ratings = pd.read_csv('../input/the-movies-dataset/ratings_small.csv')
df_movies=pd.read_csv('../input/the-movies-dataset/movies_metadata.csv')


# In[ ]:


df_movies.columns


# In[ ]:


df_movies.shape


# In[ ]:


df_movies.head()


# Note that in this dataset movies are rated on a scale of 5 unlike the earlier one.

# In[ ]:


ratings.head()


# In[ ]:


data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
data.split(n_folds=5)


# In[ ]:


svd = SVD()
evaluate(svd, data, measures=['RMSE'])


# We get a mean Root Mean Sqaure Error of 0.89 approx which is more than good enough for our case. Let us now train on our dataset and arrive at predictions.

# In[ ]:


trainset = data.build_full_trainset()
svd.fit(trainset)


# Let us pick user with user Id 1  and check the ratings she/he has given.

# In[ ]:


ratings[ratings['userId'] == 1]


# In[ ]:


df_movies.loc[df_movies['id'] > '300']


# In[ ]:


svd.predict(1, 1339, 3).est


# For movie with ID 302, we get an estimated prediction of **2.618**. 
