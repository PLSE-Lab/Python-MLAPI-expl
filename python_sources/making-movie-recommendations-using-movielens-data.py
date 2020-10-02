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
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ## Recommendation Systems
# 
# - Knowing "What customers are most likely to buy in future" is key to personalized marketing for most of the businesses. Understanding customers past purchase behavior or customer demographics could be key to make future buy predictions. But how to use the customer behavior data, depends on many different algorithms or techniques. Some alogorithms may use demographic information to make this predictions. But most of the times, the orgranizations may not have these kind of information about customers at all. All that organization will have are what customers bought in past or if the liked it or not.
# 
# - Recommendation systems use techniques to leverage these information and make recommendation, which has been proved to be very successful. For examples, Amazon.com's most popular feature of "Customers who bought this also buys this?"
# 
# - Some of the key techiques that recommendation systems use are
# 
#         - Association Rules mining
#         - Collaborative Filtering
#         - Matrix Factorization
#         - Page Rank Algorithm
# - We will discuss Collaborative filtering techinque in this article.
# - Two most widely used Collaborative filtering techniques are
# 
#         - User Similarity
#         - Item Similarity
#         
# - Here is a nice blog explanation of collaborative filtering.
# - For the purpose of demonstration, we will use the data provided by movilens. It is available here.
# - The dataset contains information about which user watched which movie and what ratings (on a scale of 1 - 5 ) he have given to the movie.

# In[ ]:


import pandas as pd
import numpy as np


# **Loading Ratings dataset**

# In[ ]:


rating_df = pd.read_csv( "../input/u.data", delimiter = "\t", header = None )


# In[ ]:


rating_df.head( 10 )


# **Name the columns**

# In[ ]:


rating_df.columns = ["userid", "movieid", "rating", "timestamp"]


# In[ ]:


rating_df.head( 10 )


# **Number of unique users**

# In[ ]:


len( rating_df.userid.unique() )


# **Number of unique movies**

# In[ ]:


len( rating_df.movieid.unique() )


# * So a total of 1682 movies and 943 users data is available in the dataset. Let's drop the timestamp columns. We do not need it.

# In[ ]:


rating_df.drop( "timestamp", inplace = True, axis = 1 )


# In[ ]:


rating_df.head( 10 )


# **Loading Movies Data**

# In[ ]:


movies_df = pd.read_csv( "../input/u.item", delimiter = '\|', header = None )


# In[ ]:


movies_df = movies_df.iloc[:,:2]
movies_df.columns = ['movieid', 'title']


# In[ ]:


movies_df.head( 10 )


# **Finding User Similarities**

# In[ ]:


from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine, correlation


# **Create the pivot table**

# In[ ]:


user_movies_df = rating_df.pivot( index='userid', columns='movieid', values = "rating" ).reset_index(drop=True)


# **Fill '0' for ratings not given by users**

# In[ ]:


user_movies_df.fillna(0, inplace = True)


# In[ ]:


user_movies_df.shape


# In[ ]:


user_movies_df.iloc[10:20, 20:30]


# ## Calculate the distances
# Based on what users have given ratings to different items, we can calculate the distances between them. Less the distance more similar they are.
# 
# For example, following users have given different ratings to differnt books.
# 
# Now, we can find similar users based the distance between user depending on how they have rated the movies. The dimensions are the books and scale is the ratings users have provided.

# In[ ]:


from IPython.display import Image
Image("../input/recomend-1.png")


# For calculating distances, many similarity coefficients can be calculated. Most widely used similarity coefficients are Euclidean, Cosine, Pearson Correlation etc.
# We will use cosine distance here. Here we are insterested in similarity. That means higher the value more similar they are. But as the function gives us the distance, we will deduct it from 1.

# In[ ]:


from IPython.display import Image
Image("../input/recomend-2.png")


# In[ ]:


user_sim = 1 - pairwise_distances( user_movies_df.as_matrix(), metric="cosine" )


# In[ ]:


user_sim_df = pd.DataFrame( user_sim )


# In[ ]:


user_sim_df[0:5]


# **Who is similar to who?**
# Users with highest similarity values can be treated as similar users.

# In[ ]:


user_sim_df.idxmax(axis=1)[0:5]


# The above results show that user are most similar to themselves. But this is not what we want. So, we will fill the diagonal of the matrix (which represent the relationship with self) with 0.
# 
# **Setting correlation with self to 0**

# In[ ]:


np.fill_diagonal( user_sim, 0 )


# In[ ]:


user_sim_df = pd.DataFrame( user_sim )


# In[ ]:


user_sim_df[0:5]


# **Finding user similarities**

# In[ ]:


user_sim_df.idxmax(axis=1).sample( 10, random_state = 10 )


# This shows which results are similar to each other. The actual user id will be the index number + 1. That means user 545 is similar to user 757 and so on and so forth.
# 
# **Movies similar users like or dislike**
# * We can find the actual movie names and check if the similar users have rated them similarity or differently.

# In[ ]:


def get_user_similar_movies( user1, user2 ):
  common_movies = rating_df[rating_df.userid == user1].merge(rating_df[rating_df.userid == user2], on = "movieid", how = "inner" )

  return common_movies.merge( movies_df, on = 'movieid' )


# **User 310 Vs. User 247**

# In[ ]:


get_user_similar_movies( 310, 247 )


# **Challenges with User similarity**
# The challenge with calculating user similarity is the user need to have some prior purchases and should have rated them. This recommendation technique does not work for new users. The system need to wait until the user make some purchases and rates them. Only then similar users can be found and recommendations can be made. This is called cold start problem. This can be avoided by calculating item similarities based how how users are buying these items and rates them together. Here the items are entities and users are dimensions.
# 
# **Finding Item Similarity**
# Let's create a pivot table of Movies to Users The rows are movies and columns are users. And the values in the matrix are the rating for a specific movie by a specific user.

# In[ ]:


rating_mat = rating_df.pivot( index='movieid', columns='userid', values = "rating" ).reset_index(drop=True)


# **Fill with 0, where users have not rated the movies**

# In[ ]:


rating_mat.fillna( 0, inplace = True )


# In[ ]:


rating_mat.shape


# In[ ]:


rating_mat.head( 10 )


# **Calculating the item distances and similarities**

# In[ ]:


movie_sim = 1 - pairwise_distances( rating_mat.as_matrix(), metric="correlation" )


# In[ ]:


movie_sim.shape


# In[ ]:


movie_sim_df = pd.DataFrame( movie_sim )


# In[ ]:


movie_sim_df.head( 10 )


# **Finding similar movies to "Toy Story"**

# In[ ]:


movies_df['similarity'] = movie_sim_df.iloc[0]
movies_df.columns = ['movieid', 'title', 'similarity']


# In[ ]:


movies_df.head( 10 )


# In[ ]:


movies_df.sort_values(by='similarity', ascending=False)[1:10]


# That means anyone who buys Toy Story and likes it, the top 3 movies that can be recommender to him or her are Star Wars (1977), Independence Day (ID4) (1996) and Rock, The (1996)
# 
# **Utility function to find similar movies**

# In[ ]:


def get_similar_movies( movieid, topN = 5 ):
  movies_df['similarity'] = movie_sim_df.iloc[movieid -1]
  top_n = movies_df.sort_values( ["similarity"], ascending = False )[0:topN]
  print( "Similar Movies to: ", )
  return top_n


# **Similar movies to Twister**

# In[ ]:


get_similar_movies( 118 )


# **Similar movies to The Godfather**

# In[ ]:


get_similar_movies( 127, 10 )


# In[ ]:




