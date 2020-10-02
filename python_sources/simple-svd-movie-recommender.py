#!/usr/bin/env python
# coding: utf-8

# ## About SVD and Python
# [How to Calculate the Singular-Value Decomposition (SVD) from Scratch with Python](https://machinelearningmastery.com/singular-value-decomposition-for-machine-learning/)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


ratings = pd.read_csv('../input/ratings_small.csv')
ratings.head()


# In[ ]:


user_item = ratings.groupby(['userId', 'movieId'])['rating'].first().unstack(fill_value=0.0)


# In[ ]:


user_item.shape


# In[ ]:


# What are the top rated movies for user 42
user_item.loc[42].sort_values(ascending=False).head()


# In[ ]:


from scipy.sparse.linalg import svds
U, sigma, Vt = svds(user_item, k = 50)


# In[ ]:


U.shape


# In[ ]:


Vt.shape


# In[ ]:


sigma_diag_matrix=np.diag(sigma)


# In[ ]:


all_user_predicted_ratings = np.dot(np.dot(U, sigma_diag_matrix), Vt)
#all_user_predicted_ratings_demeaned = all_user_predicted_ratings +  user_ratings_mean.reshape(-1, 1)
preds_df = pd.DataFrame(all_user_predicted_ratings, columns = user_item.columns, index=user_item.index)


# In[ ]:


preds_df.shape


# In[ ]:


preds_df.head()


# In[ ]:


# Top-10 recommended movies for user 42
user_item.loc[42].sort_values(ascending=False).head(10)


# In[ ]:


movies_user_42 = user_item.loc[42]


# In[ ]:


high_rated_movies_42 = movies_user_42[movies_user_42 > 3].index


# In[ ]:


high_rated_movies_42


# In[ ]:


movies_recommended_for_42 = preds_df.loc[42]


# In[ ]:


movies_high_recommend_for_42 = movies_recommended_for_42[movies_recommended_for_42 > 3].index


# In[ ]:


movies_high_recommend_for_42


# In[ ]:


# What are the movies that have a high recommendation (> 3) but that have no rating yet
set(movies_high_recommend_for_42) - set(high_rated_movies_42)


# In[ ]:


# No strong recommendations for user 42


# In[ ]:


def get_high_recommended_movies(userId):
    movies_rated_by_user = user_item.loc[userId]
    movies_high_rated_by_user =  movies_rated_by_user[movies_rated_by_user > 4].index
    movies_recommended_for_user = preds_df.loc[userId]
    movies_high_recommend_for_user = movies_recommended_for_user[movies_recommended_for_user > 4].index
    return set(movies_high_recommend_for_user) - set(movies_high_rated_by_user)


# In[ ]:


get_high_recommended_movies(42)


# In[ ]:


get_high_recommended_movies(314)


# In[ ]:


get_high_recommended_movies(217)


# In[ ]:


# User 217 should go watching movie 1198: we expect a rating of 4.2!
preds_df.loc[217, 1198]


# In[ ]:


for user_id in range(user_item.shape[0])[1:]:
    recommend_set = list(get_high_recommended_movies(user_id))
    rating_set = map(lambda x: round(preds_df.loc[user_id, x],2), recommend_set)
    result = list(zip(recommend_set, rating_set))
    if len(recommend_set) > 0:
        print('USER_ID: {}, (FILME,RATING): {}'.format(user_id, result))    

