#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


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


# In[ ]:


rating_df.drop( "timestamp", inplace = True, axis = 1 )


# In[ ]:


rating_df.head( 10 )


# In[ ]:


movies_df = pd.read_csv( "../input/u.item", delimiter = '\|', header = None )


# In[ ]:


movies_df = movies_df.iloc[:,:2]
movies_df.columns = ['movieid', 'title']


# In[ ]:


movies_df.head( 10 )


# In[ ]:


from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine, correlation


# In[ ]:


user_movies_df = rating_df.pivot( index='userid', columns='movieid', values = "rating" ).reset_index(drop=True)


# In[ ]:


user_movies_df.fillna(0, inplace = True)


# In[ ]:


user_movies_df.shape


# In[ ]:


user_movies_df.iloc[10:20, 20:30]


# In[ ]:


user_sim = 1 - pairwise_distances( user_movies_df.as_matrix(), metric="cosine" )


# In[ ]:


user_sim_df = pd.DataFrame( user_sim )


# In[ ]:


user_sim_df[0:5]


# In[ ]:


user_sim_df.idxmax(axis=1)[0:5]


# In[ ]:


np.fill_diagonal( user_sim, 0 )


# In[ ]:


user_sim_df = pd.DataFrame( user_sim )


# In[ ]:


user_sim_df[0:5]


# **Finding user similarities**

# In[ ]:


user_sim_df.idxmax(axis=1).sample( 10, random_state = 10 )


# In[ ]:


def get_user_similar_movies( user1, user2 ):
  common_movies = rating_df[rating_df.userid == user1].merge(rating_df[rating_df.userid == user2], on = "movieid", how = "inner" )

  return common_movies.merge( movies_df, on = 'movieid' )


# **User 310 Vs. User 247**

# In[ ]:


get_user_similar_movies( 310, 247 )


# In[ ]:


rating_mat = rating_df.pivot( index='movieid', columns='userid', values = "rating" ).reset_index(drop=True)


# In[ ]:


rating_mat.fillna( 0, inplace = True )


# In[ ]:


rating_mat.shape


# In[ ]:


rating_mat.head( 10 )


# In[ ]:


movie_sim = 1 - pairwise_distances( rating_mat.as_matrix(), metric="correlation" )


# In[ ]:


movie_sim.shape


# In[ ]:


movie_sim_df = pd.DataFrame( movie_sim )


# In[ ]:


movie_sim_df.head( 10 )


# In[ ]:


movies_df['similarity'] = movie_sim_df.iloc[0]
movies_df.columns = ['movieid', 'title', 'similarity']


# In[ ]:


movies_df.head( 10 )


# In[ ]:


movies_df.sort_values(by='similarity', ascending=False)[1:10]


# In[ ]:


def get_similar_movies( movieid, topN = 5 ):
  movies_df['similarity'] = movie_sim_df.iloc[movieid -1]
  top_n = movies_df.sort_values( ["similarity"], ascending = False )[0:topN]
  print( "Similar Movies to: ", )
  return top_n


# **Similar movies to Twister**

# In[ ]:


get_similar_movies( 118, 10 )


# **Similar movies to The Godfather**

# In[ ]:


get_similar_movies( 127, 10 )

