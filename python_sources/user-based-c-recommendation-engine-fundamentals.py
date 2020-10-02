#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/"))

# Any results you write to the current directory are saved as output.


# ### Loading and Reading Dataset

# In[ ]:


ratings  = pd.read_csv('../input/ratings_small.csv')
ratings.head()


# In[ ]:


print(ratings.shape)


# ## Splitting Data into Train and Test Set

# In[ ]:


from sklearn.model_selection import train_test_split
train_df,test_df = train_test_split(ratings, test_size = 0.3, random_state = 42)
print(train_df.shape, '\t\t', test_df.shape)


# In[ ]:


train_df.head()


# #### As MovieID column stores, movies and our first task is to build a recommendation engine based on USER COLLABORATIVE approach, we want our data in tabular format such that: userID as index, Data i.e Unique MovieID's as Columns/features and Ratings as Values

# ### We can achieve this using dataframe's Pivot Method

# In[ ]:


df_movies_as_features = train_df.pivot(index = 'userId', columns = 'movieId',values = 'rating' )
df_movies_as_features.shape


# In[ ]:


df_movies_as_features.head()


# In[ ]:


df_movies_as_features.fillna(0, inplace = True)
df_movies_as_features.head()


# In[ ]:





#   ####  Copy Train and Test DATASET
#   This will be used for Evaluation and

# In[ ]:


dummy_train=train_df.copy()
dummy_test =test_df.copy()


# In[ ]:


dummy_train.rating.value_counts()


# In[ ]:


dummy_test.rating.value_counts()


# In[ ]:


dummy_train['rating'] =dummy_train.rating.apply(lambda x : 0  if x >=1 else 1)
dummy_test['rating'] =dummy_test.rating.apply(lambda x : 1  if x >=1 else 0)
dummy_train.head()


# In[ ]:


def pivot_by_movie(df):
    df = df.pivot(index='userId', columns = 'movieId', values = 'rating')
    df.fillna(0, inplace = True)
    return df


# In[ ]:


dummy_train = pivot_by_movie(dummy_train)
dummy_test = pivot_by_movie(dummy_test)


# In[ ]:


dummy_train.head()


# Note: Movie No 10 userid 2 has rating as 1 in dummy_train as this movie was not reviewed by this particular user. See  Movie No 10 userid 2 has rating as 1 in Features_movie_df

# ### Let's create User Similarity Matrix
# Using Cosine Similarities= dot_product of each row with Other rows to calculate simialarity scores between users

# 

# In[ ]:


from sklearn.metrics.pairwise import pairwise_distances


# In[ ]:


user_correlation  = 1- pairwise_distances(df_movies_as_features,metric = 'cosine')
user_correlation.shape


# In[ ]:


user_correlation


# In[ ]:


np.sum(np.isnan(user_correlation))


# In[ ]:


user_correlation[np.isnan(user_correlation)]=0
user_correlation


# ### But to find cosine similarity we have considered all the rating but 0 should have been not Considered for calculating similarity.
# So let's load the Training data again and not replace NaN with zero 

# In[ ]:


train_movies_as_feature = train_df.pivot(index='userId', columns = 'movieId', values = 'rating')
train_movies_as_feature.head()


# In[ ]:


train_movies_as_feature.shape


# let's Normalize it as well

# In[ ]:


mean = np.nanmean(train_movies_as_feature, axis = 1)
print(mean.shape)


# As we calculated mean row wise let's deduct mean rating from each value and normalize by taking transpose 

# In[ ]:


normalised_df = (train_movies_as_feature.T-mean).T
normalised_df.head()


# #### Why did we Normalized it?
# The Reason being a lot of guys give high scores to even bad movies and very Good score to Good movies while others give a very low score to bad ones. So it would be better we subtracted the mean score given by each guy to all movies from score given by that guy to each movie.

# ### Let's calculate cosine similarity between users again

# In[ ]:


user_correlation =  1 - pairwise_distances(normalised_df.fillna(0), metric ='cosine')
user_correlation[np.isnan(user_correlation)]=0
print(user_correlation)


# **In this matrix each column corresponds to One user and values in column corresponds to how is he correlated with other users. 
# For example, let's take 1st column. the value 1 signifies he is 100% related to user 1 (obviously as he is himself User 1) while next value signifies how much he is related to second user**

# In[ ]:


user_correlation.shape


# #### Before we use User Correlation and start Calcuting predicted ratings we have to remove user who are negatively correlated as we don't need them.
# We want only users who are similar for our predicting Ratings

# In[ ]:


user_correlation[user_correlation<0]=0


# ### Now we need to calculate how much rating a user will give to Movies that he has not Rated. For this purpose we will use a dot product on user_correlation with movie_score_feature_matrix so that user's correlation with him and others will be used to calulate average user score he should have given to Movies.

# In[ ]:


user_predicted_ratings = np.dot(user_correlation, train_movies_as_feature.fillna(0))
user_predicted_ratings


# In[ ]:


user_predicted_ratings.shape


# Now we have predicted ratings for each movie by a user represented by each row in one columns

# Now we have to remove predicted scores for movies which the user himself has already given score i.e He has watched.
# 
# This is where Dummy_train and Test come in picture.
# We will use element wise multiplication of predcited_score_matrix with Dummy_train and Dummy_test in order to Nullify movies which has already been watched

# In[ ]:


user_final_rating = np.multiply(user_predicted_ratings, dummy_train)
user_final_rating.head()


# #### Finding Top 5 recommendation for User 1

# In[ ]:


user_final_rating.iloc[670].sort_values(ascending =False)[-6:]


# > **Similary we can find Ratings for other users as well.
# > This marks the end of User Based Collaborative Filtering.**

# In[ ]:




