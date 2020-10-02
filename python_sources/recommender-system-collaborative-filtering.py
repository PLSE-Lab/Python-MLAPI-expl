#!/usr/bin/env python
# coding: utf-8

# 
# Recommender systems are among the most popular applications of data science today. They are used to predict the "rating" or "preference" that a user would give to an item. Almost every major tech company has applied them in some form or the other: Amazon uses it to suggest products to customers, YouTube uses it to decide which video to play next on autoplay, and Facebook uses it to recommend pages to like and people to follow.

# Broadly, recommender systems can be classified into 3 types:
# 
#     Simple recommenders: offer generalized recommendations to every user, based on movie popularity and/or genre. The basic idea behind this system is that movies that are more popular and critically acclaimed will have a higher probability of being liked by the average audience. IMDB Top 250 is an example of this system.
#     Content-based recommenders: suggest similar items based on a particular item. This system uses item metadata, such as genre, director, description, actors, etc. for movies, to make these recommendations. The general idea behind these recommender systems is that if a person liked a particular item, he or she will also like an item that is similar to it.
#     Collaborative filtering engines: these systems try to predict the rating or preference that a user would give an item-based on past ratings and preferences of other users. Collaborative filters do not require item metadata like its content-based counterparts.
# 

# This is a basic model of collabrative filtering.

# In[17]:


import numpy as np # linear algebra
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
ratings=pd.read_csv('../input/ratings.csv')
ratings.head(15)


# In[18]:


movies=pd.read_csv('../input/movies.csv')
movies.head(15)


# In[19]:


movie_ratings = pd.merge(movies, ratings)
movie_ratings.head(15)


# In[20]:


ratings_matrix = ratings.pivot_table(index=['movieId'],columns=['userId'],values='rating').reset_index(drop=True)
ratings_matrix.fillna( 0, inplace = True )
ratings_matrix.head(15)


# In[21]:


movie_similarity=cosine_similarity(ratings_matrix)
np.fill_diagonal( movie_similarity, 0 ) 
movie_similarity


# In[22]:


ratings_matrix = pd.DataFrame( movie_similarity )
ratings_matrix.head(15)


# In[24]:


try:
    #user_inp=input('Enter the reference movie title based on which recommendations are to be made: ')
    user_inp="Jumanji (1995)"
    inp=movies[movies['title']==user_inp].index.tolist()
    inp=inp[0]
    
    movies['similarity'] = ratings_matrix.iloc[inp]
    movies.head(5)
    
except:
    print("Sorry, the movie is not in the database!")
    
print("Recommended movies based on your choice of ",user_inp ,": \n", movies.sort_values( ["similarity"], ascending = False )[1:10])

