#!/usr/bin/env python
# coding: utf-8

# <h3>**Movie Recommendation using Collaborative Filtering**</h3>
# 
# Also known as User-User Filtering. 
# 
# It uses other users to recommend items to the input user. It attempts to find users that have similar preferences and opinions as the input and then recommends items that they have liked to the input. There are several methods of finding similar users (Even some making use of Machine Learning), and the one we will be using here is going to be based on the Pearson Correlation Function.
# 
# 
# 
# 
# **Please upvote if you find this helpful**
# 

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


# In[ ]:


import numpy as np
import pandas as pd
from math import sqrt
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# Reading the datasets that we have. We will need movies and the ratings files. 

# In[ ]:


movie = pd.read_csv('/kaggle/input/movielens-20m-dataset/movie.csv')
rating= pd.read_csv('/kaggle/input/movielens-20m-dataset/rating.csv')


# Quick view of the two dataframes, the rows and the columns

# In[ ]:


movie.head()


# In[ ]:


rating.head()


# Using pandas' replace function, we remove the year from the title in the movie dataframe and add a separate year column.
# This is done using regular expressions. Find a year stored between parentheses. We specify the parantheses so we don't conflict with movies that have years in their titles.

# First we create the new Year column in the movie dataframe.

# In[ ]:


movie['year'] = movie.title.str.extract('(\(\d\d\d\d\))',expand=False)
#Removing the parentheses
movie['year'] = movie.year.str.extract('(\d\d\d\d)',expand=False)


# Now removing the year from the title in the Title column. 

# In[ ]:


#Removing the years from the 'title' column
movie['title'] = movie.title.str.replace('(\(\d\d\d\d\))', '')
#Applying the strip function to get rid of any ending whitespace characters that may have appeared
movie['title'] = movie['title'].apply(lambda x: x.strip())


# In[ ]:


movie.head()


# Collaborative filtering doesn't recommend based on the features of the movie. The recommendation is based on the likes and dislikes or ratings of the neighbours or other users. 
# So we will drop the genre column, since there is no use of it. 

# In[ ]:


movie.drop(columns=['genres'], inplace=True)


# In[ ]:


movie.head()


# Now, coming to the ratings dataframe, we have the movieId column that is common with the movie dataframe. Each user has given multiple ratings for different movies. The column Timestamp is not needed for the recommendation system. So we can drop it. 

# In[ ]:


rating.drop(columns=['timestamp'],inplace=True)


# In[ ]:


rating.head()


# The technique is called Collaborative Filtering, which is also known as User-User Filtering. As hinted by its alternate name, this technique uses other users to recommend items to the input user. It attempts to find users that have similar preferences and opinions as the input and then recommends items that they have liked to the input. There are several methods of finding similar users (Even some making use of Machine Learning), and the one we will be using here is going to be based on the Pearson Correlation Function.

# The process for creating a User Based recommendation system is as follows:
# - Select a user with the movies the user has watched
# - Based on his rating to movies, find the top X neighbours 
# - Get the watched movie record of the user for each neighbour.
# - Calculate a similarity score using some formula
# - Recommend the items with the highest score

# In[ ]:


user = [
            {'title':'Breakfast Club, The', 'rating':4},
            {'title':'Toy Story', 'rating':2.5},
            {'title':'Jumanji', 'rating':3},
            {'title':"Pulp Fiction", 'rating':4.5},
            {'title':'Akira', 'rating':5}
         ] 
inputMovie = pd.DataFrame(user)
inputMovie


# We need to now add the movieId column from the movie dataframe into the inputMovie Dataframe. 
# First filter out the rows that contain the input movies' title and then merging this subset with the input dataframe. We also drop unnecessary columns for the input to save memory space.

# In[ ]:


#Filtering out the movies by title
Id = movie[movie['title'].isin(inputMovie['title'].tolist())]
#Then merging it so we can get the movieId. It's implicitly merging it by title.
inputMovie = pd.merge(Id, inputMovie)
#Dropping information we won't use from the input dataframe
inputMovie = inputMovie.drop('year', 1)
inputMovie


# **Finding the users who have seen the same movies from the rating dataframe**
# With the movie ID's in our input, we can now get the subset of users that have watched and reviewed the movies in our input.

# In[ ]:


#Filtering out users that have watched movies that the input has watched and storing it
users = rating[rating['movieId'].isin(inputMovie['movieId'].tolist())]
users.head()


# In[ ]:


users.shape


# In[ ]:


#Groupby creates several sub dataframes where they all have the same value in the column specified as the parameter
userSubsetGroup = users.groupby(['userId'])


# In[ ]:


#showing one such group example by getting all the users of a particular uderId
userSubsetGroup.get_group(1130)


# In[ ]:


#Sorting it so users with movie most in common with the input will have priority
userSubsetGroup = sorted(userSubsetGroup,  key=lambda x: len(x[1]), reverse=True)


# In[ ]:


userSubsetGroup[0:3]


# **Similarity of users to input user**
# Next, we are going to compare all users  to our specified user and find the one that is most similar.
# we're going to find out how similar each user is to the input through the Pearson Correlation Coefficient. It is used to measure the strength of a linear association between two variables. The formula for finding this coefficient between sets X and Y with N values can be seen in the image below.
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

# In[ ]:


userSubsetGroup = userSubsetGroup[0:100]


# In[ ]:


#Store the Pearson Correlation in a dictionary, where the key is the user Id and the value is the coefficient
pearsonCorDict = {}

#For every user group in our subset
for name, group in userSubsetGroup:
    #Let's start by sorting the input and current user group so the values aren't mixed up later on
    group = group.sort_values(by='movieId')
    inputMovie = inputMovie.sort_values(by='movieId')
    #Get the N for the formula
    n = len(group)
    #Get the review scores for the movies that they both have in common
    temp = inputMovie[inputMovie['movieId'].isin(group['movieId'].tolist())]
    #And then store them in a temporary buffer variable in a list format to facilitate future calculations
    tempRatingList = temp['rating'].tolist()
    #put the current user group reviews in a list format
    tempGroupList = group['rating'].tolist()
    #Now let's calculate the pearson correlation between two users, so called, x and y
    Sxx = sum([i**2 for i in tempRatingList]) - pow(sum(tempRatingList),2)/float(n)
    Syy = sum([i**2 for i in tempGroupList]) - pow(sum(tempGroupList),2)/float(n)
    Sxy = sum( i*j for i, j in zip(tempRatingList, tempGroupList)) - sum(tempRatingList)*sum(tempGroupList)/float(n)
    
    #If the denominator is different than zero, then divide, else, 0 correlation.
    if Sxx != 0 and Syy != 0:
        pearsonCorDict[name] = Sxy/sqrt(Sxx*Syy)
    else:
        pearsonCorDict[name] = 0


# In[ ]:


pearsonCorDict.items()


# In[ ]:


pearsonDF = pd.DataFrame.from_dict(pearsonCorDict, orient='index')
pearsonDF.columns = ['similarityIndex']
pearsonDF['userId'] = pearsonDF.index
pearsonDF.index = range(len(pearsonDF))
pearsonDF.head()


# In[ ]:


topUsers=pearsonDF.sort_values(by='similarityIndex', ascending=False)[0:50]
topUsers.head()


# **Rating of selected users to all movies**
# We're going to do this by taking the weighted average of the ratings of the movies using the Pearson Correlation as the weight. But to do this, we first need to get the movies watched by the users in our pearsonDF from the ratings dataframe and then store their correlation in a new column called _similarityIndex". This is achieved below by merging of these two tables.

# In[ ]:


topUsersRating=topUsers.merge(rating, left_on='userId', right_on='userId', how='inner')
topUsersRating.head()


# multiply the movie rating by its weight (The similarity index), then sum up the new ratings and divide it by the sum of the weights.
# 
# We can easily do this by simply multiplying two columns, then grouping up the dataframe by movieId and then dividing two columns:

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


# In[ ]:


recommendation_df = recommendation_df.sort_values(by='weighted average recommendation score', ascending=False)
recommendation_df.head(10)


# In[ ]:


movie.loc[movie['movieId'].isin(recommendation_df.head(10)['movieId'].tolist())]


# In[ ]:




