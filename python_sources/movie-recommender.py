#!/usr/bin/env python
# coding: utf-8

# In this notebook, a simple implementation of a recommendation systems based on Collaborative Filtering for movies is presented.

# <h1>Table of contents</h1>
# 
# <div class="alert alert-block alert-info" style="margin-top: 20px">
#     <ol>
#         <li><a href="#ref2">Preprocessing</a></li>
#         <li><a href="#ref3">Collaborative Filtering</a></li>
#     </ol>
# </div>
# <br>
# <hr>

# In[ ]:


import pandas as pd
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# <hr>
# 
# <a id="ref2"></a>
# # Preprocessing

# data used: 
# Full: 27,000,000 ratings and 1,100,000 tag applications applied to 58,000 movies by 280,000 users. Includes tag genome data with 14 million relevance scores across 1,100 tags. Last updated 9/2018.
# from https://grouplens.org/datasets/movielens/

# In[ ]:


movies_df = pd.read_csv('/kaggle/input/grouplens-2018/ml-latest/movies.csv')
ratings_df = pd.read_csv('/kaggle/input/grouplens-2018/ml-latest/ratings.csv')


# In[ ]:


movies_df.shape


# In[ ]:


movies_df.tail()


# In[ ]:


#Using regular expressions to find a year stored between parentheses
#We specify the parantheses so we don't conflict with movies that have years in their titles
movies_df['year'] = movies_df.title.str.extract('(\(\d\d\d\d\))',expand=False)
#Removing the parentheses
movies_df['year'] = movies_df.year.str.extract('(\d\d\d\d)',expand=False)
#Removing the years from the 'title' column
movies_df['title'] = movies_df.title.str.replace('(\(\d\d\d\d\))', '')
#Applying the strip function to get rid of any ending whitespace characters that may have appeared
movies_df['title'] = movies_df['title'].apply(lambda x: x.strip())
movies_df.head()


# In[ ]:


#Dropping the genres column, no need for them
movies_df = movies_df.drop('genres', 1)
movies_df.head()


# In[ ]:


ratings_df.head()


# In[ ]:


#Drop removes a specified row or column from a dataframe
ratings_df = ratings_df.drop('timestamp', 1)
ratings_df.head()


# <hr>
# 
# <a id="ref3"></a>
# # Collaborative Filtering

# The process for creating a User Based recommendation system is as follows:
# - Select a user with the movies the user has watched
# - Based on his rating to movies, find the top X neighbours 
# - Get the watched movie record of the user for each neighbour.
# - Calculate a similarity score using some formula
# - Recommend the items with the highest score

# In[ ]:


# here's a hypothetical user that we want to make suggestions for
userInput = [
            {'title':'Avatar 2', 'rating':7},
            {'title':'13 Hours', 'rating':3.5},
            {'title':'Jumanji', 'rating':7},
            {'title':"Sherlock: The Abominable Bride", 'rating':8},
            {'title':'Jurassic World', 'rating':8},
    {'title':'Star Wars: Episode VII - The Force Awakens', 'rating':6},
    {'title':'Avengers: Age of Ultron', 'rating':9},
    {'title':'Ant-Man', 'rating':8},
    {'title':'Justice League: Throne of Atlantis', 'rating':7}]
inputMovies = pd.DataFrame(userInput)
inputMovies


# #### Add movieId to input user

# In[ ]:


inputId = movies_df[movies_df['title'].isin(inputMovies['title'].tolist())]
inputId.head()


# In[ ]:


inputMovies = pd.merge(inputId, inputMovies)
inputMovies


# In[ ]:


inputMovies = inputMovies.drop('year', 1)
inputMovies


# #### The users who has seen the same movies

# In[ ]:


#Filtering out users that have watched movies that the input has watched and storing it
userSubset = ratings_df[ratings_df['movieId'].isin(inputMovies['movieId'].tolist())]
userSubset.head()


# We now group up the rows by user ID.

# In[ ]:


userSubsetGroup = userSubset.groupby(['userId'])


# In[ ]:


#Sorting it so users with movie most in common with the input will have priority
userSubsetGroup = sorted(userSubsetGroup,  key=lambda x: len(x[1]), reverse=True)


# Now lets look at the first user

# In[ ]:


userSubsetGroup[0]


# #### Similarity of users to input user
# Next, we are going to compare all users (not really all !!!) to our specified user and find the one that is most similar.  
# we're going to find out how similar each user is to the input through the __Pearson Correlation Coefficient__. It is used to measure the strength of a linear association between two variables. The formula for finding this coefficient between sets X and Y with N values can be seen in the image below. 
# 
# Why Pearson Correlation?
# 
# Pearson correlation is invariant to scaling, i.e. multiplying all elements by a nonzero constant or adding any constant to all elements. For example, if you have two vectors X and Y,then, pearson(X, Y) == pearson(X, 2 * Y + 3). This is a pretty important property in recommendation systems because for example two users might rate two series of items totally different in terms of absolute rates, but they would be similar users (i.e. with similar ideas) with similar rates in various scales .
# 
# ![alt text](https://wikimedia.org/api/rest_v1/media/math/render/svg/bd1ccc2979b0fd1c1aec96e386f686ae874f9ec0 "Pearson Correlation")
# 
# The values given by the formula vary from r = -1 to r = 1, where 1 forms a direct correlation between the two entities (it means a perfect positive correlation) and -1 forms a perfect negative correlation. 
# 
# In our case, a 1 means that the two users have similar tastes while a -1 means the opposite.

# We will select a subset of users to iterate through. This limit is imposed because we don't want to waste too much time going through every single user.

# In[ ]:


userSubsetGroup = userSubsetGroup[0:100]


# Now, we calculate the Pearson Correlation between input user and subset group, and store it in a dictionary, where the key is the user Id and the value is the coefficient
# 

# In[ ]:


#Store the Pearson Correlation in a dictionary, where the key is the user Id and the value is the coefficient
pearsonCorrelationDict = {}

#For every user group in our subset
for name, group in userSubsetGroup:
    #Let's start by sorting the input and current user group so the values aren't mixed up later on
    group = group.sort_values(by='movieId')
    inputMovies = inputMovies.sort_values(by='movieId')
    #Get the N for the formula
    nRatings = len(group)
    #Get the review scores for the movies that they both have in common
    temp_df = inputMovies[inputMovies['movieId'].isin(group['movieId'].tolist())]
    #And then store them in a temporary buffer variable in a list format to facilitate future calculations
    tempRatingList = temp_df['rating'].tolist()
    #Let's also put the current user group reviews in a list format
    tempGroupList = group['rating'].tolist()
    #Now let's calculate the pearson correlation between two users, so called, x and y
    Sxx = sum([i**2 for i in tempRatingList]) - pow(sum(tempRatingList),2)/float(nRatings)
    Syy = sum([i**2 for i in tempGroupList]) - pow(sum(tempGroupList),2)/float(nRatings)
    Sxy = sum( i*j for i, j in zip(tempRatingList, tempGroupList)) - sum(tempRatingList)*sum(tempGroupList)/float(nRatings)
    
    #If the denominator is different than zero, then divide, else, 0 correlation.
    if Sxx != 0 and Syy != 0:
        pearsonCorrelationDict[name] = Sxy/sqrt(Sxx*Syy)
    else:
        pearsonCorrelationDict[name] = 0


# In[ ]:


pearsonDF = pd.DataFrame.from_dict(pearsonCorrelationDict, orient='index')
pearsonDF.columns = ['similarityIndex']
pearsonDF['userId'] = pearsonDF.index
pearsonDF.index = range(len(pearsonDF))
pearsonDF.head()


# #### The top x similar users to input user
# Now let's get the top 50 users that are most similar to the input.

# In[ ]:


topUsers=pearsonDF.sort_values(by='similarityIndex', ascending=False)[0:50]
topUsers.head()


# Now, let's start recommending movies to the input user.
# 
# #### Rating of selected users to all movies
# We're going to do this by taking the weighted average of the ratings of the movies using the Pearson Correlation as the weight. But to do this, we first need to get the movies watched by the users in our __pearsonDF__ from the ratings dataframe and then store their correlation in a new column called _similarityIndex". This is achieved below by merging of these two tables.

# In[ ]:


topUsersRating = topUsers.merge(ratings_df, left_on='userId', right_on='userId', how='inner')
topUsersRating.head()


# Now all we need to do is simply multiply the movie rating by its weight (The similarity index), then sum up the new ratings and divide it by the sum of the weights.
# 
# We can easily do this by simply multiplying two columns, then grouping up the dataframe by movieId and then dividing two columns:
# 
# It shows the idea of all similar users to candidate movies for the input user:

# In[ ]:


#Multiplies the similarity by the user's ratings
topUsersRating['weightedRating'] = topUsersRating['similarityIndex']*topUsersRating['rating']
topUsersRating.head()


# In[ ]:


#Applies a sum to the topUsers after grouping it up by userId
tempTopUsersRating = topUsersRating.groupby('movieId').sum()[['similarityIndex','weightedRating']]
tempTopUsersRating.columns = ['sum_similarityIndex','sum_weightedRating']
tempTopUsersRating.head()


# In[ ]:


#Creates an empty dataframe
recommendation_df = pd.DataFrame()
#Now we take the weighted average
recommendation_df['weighted average recommendation score'] = tempTopUsersRating['sum_weightedRating']/tempTopUsersRating['sum_similarityIndex']
recommendation_df['movieId'] = tempTopUsersRating.index
recommendation_df.head()


# Now let's sort it and see the top 20 movies that the algorithm recommended!

# In[ ]:


recommendation_df = recommendation_df.sort_values(by='weighted average recommendation score', ascending=False)
recommendation_df.head(10)


# In[ ]:


movies_df.loc[movies_df['movieId'].isin(recommendation_df.head(10)['movieId'].tolist())]


# In[ ]:




