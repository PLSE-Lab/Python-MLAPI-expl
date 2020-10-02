#!/usr/bin/env python
# coding: utf-8

# # Content based recommender system
# 
# Content based recommnder system works based on the genre of the movie and rating provided by user the for movies.
# 
# Suppose user has rated movies as following
# 
# |Movie Name  | Genres   | Rating | 
# |------------- | ------------- | ------|
# |Movie A  | Drama, Action, Commedy |  4 |
# |Movie B  | Comedy, Adenture | 5 |
# 
# We encode movies with one hot encoding schema
# 
# 
# |         | Drama | Action  | Comedy | Adventure |
# |---------|-------|---------|--------|-----------|
# | Movie A | 1     | 1       | 1      | 0         |
# | Movie B | 0     | 0       | 1      | 1         |
# 
# 
# We calculate weighted feature matrix by multiplying ratings to Weighted genre matrix
# 
# |         | Drama | Action  | Comedy | Adventure |
# |---------|-------|---------|--------|-----------|
# | Movie A | 4     | 4       | 4      | 0         |
# | Movie B | 0     | 0       | 5      | 5         |
# 
# Create user profile by taking sum along columns
# 
# | Drama | Action  | Comedy | Adventure |
# |---------|-------|---------|--------|
# | 4     | 4       | 9      | 5         |
# 
# 
# Create Normalised user profile
# 
# | Drama | Action  | Comedy | Adventure |
# |-------|---------|--------|-----------|
#  | 0.18  | 0.18    | 0.41   | 0.23      |
#  
# We have following movies to recommend to user with genre
# 
# |         | Drama | Action  | Comedy | Adventure |
# |---------|-------|---------|--------|-----------|
# | Movie C | 1     | 0       | 0      | 1         |
# | Movie D | 0     | 1       | 1      | 0         |
# | Movie E | 1     | 0       | 0      | 0         |
# 
# We will multiply user profile with this matrix and take sum along row
# 
# |         | Drama | Action  | Comedy | Adventure | Total 
# |---------|-------|---------|--------|-----------| ------|
# | Movie C | 0.18  | 0       | 0      | 0.23      | 0.41  |
# | Movie D | 0     | 0.18    | 0.41   | 0         | 0.59 |
# | Movie E | 0.18  | 0       | 0      | 0         | 0.18 |
# 
# This is our recommendation matrix. We can recommend Movie D to user

# ## Download Data from MovieLens
# Data can be downloaded from MovieLens database http://files.grouplens.org/datasets/movielens/ml-1m.zip

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


# ## Import required libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")


# ## Import dataset

# In[ ]:


movies_df = pd.read_csv('/kaggle/input/movies/movies.csv', sep='::', names=['MovieID','Title','Genres'])
ratings_df = pd.read_csv('/kaggle/input/ratings/ratings.csv', sep='::', names=['UserID','MovieID','Rating','Timestamp'])
users_df = pd.read_csv('/kaggle/input/users/users.csv', sep='::',names=['UserID','Gender','Age','Occupation','Zip-code'])


# ## Let's take a quick look at our data

# In[ ]:


print('Shape of movies dataset {}'.format(movies_df.shape))
print('Shape of ratings dataset {}'.format(ratings_df.shape))
print('Shape of users dataset {}'.format(users_df.shape))


# In[ ]:


movies_df.head()


# In[ ]:


ratings_df.head()


# In[ ]:


users_df.head()


# ## Preprocessing of Movies dataframe
# 

# Extracting year from title and creating separate column
# 

# In[ ]:


movies_df['Year'] = movies_df['Title'].str.extract('(\(\d\d\d\d\))')
movies_df['Year'] = movies_df['Year'].str.extract('(\d\d\d\d)')


# Now as we have extracted year and created separate column, we can remove it from title

# In[ ]:


movies_df['Title'] = movies_df['Title'].str.replace('(\(\d\d\d\d\))','')


# Remove white spaces from the beginning and end of the Title

# In[ ]:


movies_df['Title'] = movies_df['Title'].apply(lambda title : title.strip())


# ## Processing Genres columns
# 

# We will split Genres on '|' and create list with python split function
# 

# In[ ]:


movies_df['Genres'] = movies_df['Genres'].apply(lambda genres : genres.split('|'))


# In[ ]:


movies_df.head()


# We will create copy for movies dataframe and create separate column for each genre with one hot encoding

# In[ ]:


moviesWithGenres_df = movies_df.copy()


# In[ ]:


for index, row in movies_df.iterrows():
    for genre in row['Genres']:
        moviesWithGenres_df.at[index, genre] = 1
        


# In[ ]:


moviesWithGenres_df.head()


# Replace NaN values with 0

# In[ ]:


moviesWithGenres_df.fillna(0, inplace=True)


# ## Preprocessing Rating dataframe

# In[ ]:


ratings_df.head()


# We don't need Timestamp column. Let's drop it

# In[ ]:


ratings_df.drop('Timestamp', axis=1, inplace=True)


# ## Getting movies for particular user
# 
# Let's get movies for user with ID 12

# In[ ]:


user_12_ratings = ratings_df[ratings_df['UserID'] == 12]
user_12_ratings.head()


# In order to create user profile we will need genre of the movies user has given rating. We will get this from moviesWithGenres_df dataframe

# In[ ]:


user_12_ratings = pd.merge(user_12_ratings, moviesWithGenres_df, on='MovieID')
user_12_ratings.head()


# To create user profile, we need only genre columns. So drop other columns

# In[ ]:


user_12_genre = user_12_ratings.drop(columns=['MovieID', 'Rating','UserID','Title','Genres','Year'], axis=1)
user_12_genre


# We will take dot product of ratings given by each movie with genre and find out what kind of genre user likes most

# In[ ]:


user_12_profile = user_12_genre.transpose().dot(user_12_ratings['Rating'])


# In[ ]:


plt.figure(figsize=(10,8))
sns.barplot(data= user_12_profile.reset_index().sort_values(by=0, ascending =False), x = 'index', y=0)
plt.title('Genres preferred by user 12', fontsize=24)
plt.xticks(rotation=90, fontsize=12)
plt.ylabel('Percentage',fontsize=16)
plt.xlabel('Genre',fontsize=16)


# ### Most preferred Genre of user is Drame, followed by Comedy, action and crime

# #### In order to recommend movies to user we will create genre table of all movies

# Drop unnecessary columns

# In[ ]:


genre_table = moviesWithGenres_df.drop(columns=['Title', 'Genres','Year','MovieID'],axis=1)
genre_table.head()


# Multiply user profile with genre table

# In[ ]:


recommendation_user_12 = genre_table * user_12_profile
recommendation_user_12.head()


# Calculate average rating for each movie

# In[ ]:


recommendation_user_12 = recommendation_user_12.sum(axis=1)/user_12_profile.sum()


# Convert movies to recommend to user 12 into dataframe

# In[ ]:


recommendation_user_12 = pd.DataFrame(recommendation_user_12)
recommendation_user_12 = recommendation_user_12.reset_index()
recommendation_user_12.rename(columns = {'index':'MovieID', 0:'Recommend_Percent'},inplace=True)
recommendation_user_12 = recommendation_user_12.sort_values(by='Recommend_Percent',ascending=False)
recommendation_user_12.head(10)


# In order to get names we will join with movies_df on MovieID column

# In[ ]:


recommendation_user_12 = pd.merge(recommendation_user_12,movies_df, on='MovieID')
recommendation_user_12.head(10)


# ## Creating functions to get recommendation
# 
# Convert above data to functions

# In[ ]:


def get_user_profile(userID):
    '''
       Input required: Id of the user
       Returns user profile in the form of pandas Series object. 
       User profile is percentage of each genre rated/liked by user
       
    '''
    userID_ratings = ratings_df[ratings_df['UserID'] == userID]
    userID_ratings = pd.merge(userID_ratings, moviesWithGenres_df, on='MovieID')
    userID_genre = userID_ratings.drop(columns=['MovieID', 'Rating','UserID','Title','Genres','Year'], axis=1)
    user_profile = userID_genre.transpose().dot(userID_ratings['Rating'])
    
    return user_profile


# In[ ]:


# test above function
get_user_profile(12)


# In[ ]:


def get_recommendation_for_user(user_ID, number_of_movies=10):
    '''
        Returns movies with recommendation percentage in the form of pandas dataframe
        
    '''
    user_profile=  get_user_profile(user_ID)
    recommendation_for_user = genre_table * user_profile
    recommendation_for_user = recommendation_for_user.sum(axis=1)/user_12_profile.sum()
    recommendation_for_user = pd.DataFrame(recommendation_for_user, columns=['Recommend_Percent'])
    recommendation_for_user.index.name='idx'
    recommendation_for_user.reset_index(inplace=True)
    recommendation_for_user.rename(columns={'idx':"MovieID"}, inplace=True)
    recommendation_for_user = recommendation_for_user.sort_values(by='Recommend_Percent',ascending=False)
    recommendation_for_user = recommendation_for_user.head(number_of_movies)
    
    recommendation_for_user = pd.merge(recommendation_for_user,movies_df, on='MovieID')
    return recommendation_for_user


# In[ ]:


# test above function for some users
get_recommendation_for_user(12,5)


# In[ ]:


get_recommendation_for_user(25)


# In[ ]:


get_recommendation_for_user(311)


# 
