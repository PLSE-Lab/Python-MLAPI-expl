#!/usr/bin/env python
# coding: utf-8

# # Movie rating prediction and recommendation
# 
# In this tutorial we are going to introduce about recommender system. Primarily, we will build a recommender system here. 

# ## What is recommender system
# A recommender system is a simple machine learning algorithm whose aim is to provide the most relevant information to a user by discovering patterns in a dataset. The algorithm rates the items and shows the user the items that they would rate highly. An example of recommendation in action is when you visit Amazon and you notice that some items are being recommended to you or when Netflix recommends certain movies to you. They are also used by Music streaming applications such as Spotify, ganna, savan, and Deezer to recommend music that you might like. 

# ## Setting the backend of matplotlib to the 'inline' backend

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# ## Importing important packages and libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

# machine learning
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# ---
# ## Loading dataset
# 
# Loading the movies and rating data set

# In[ ]:


movies=pd.read_csv('../input/movies-data/movies_metadata.csv')
ratings=pd.read_csv('../input/movies-data/ratings_small.csv')


# ## Data overview

# In[ ]:


movies.head(2)


# In[ ]:


ratings.head()


# ## Data desciption
# 
# **adult:** Give the adult rating to the movie
# 
# **belong_to_collection:** The parent directory of the movie
# 
# **budget:** The budget for the movie
# 
# **genres:** The denote a style or category of art, music, or literature like animation, adventure, etc.
# 
# **homepage:** The web address for the movie
# 
# **id:** The idebtification number for a movie
# 
# **imdb_id:** The identification number given by imdb
# 
# **original_language:** The language of the movie
# 
# **original_title:** The name of the movie
# 
# **overview:** Some explanation of the movie
# 
# **release_date:** The date of the release of the movie
# 
# **revenue:** The revenue genrated by the movie
# 
# **runtime:** The total time of the movie
# 
# **status:** The status of the movie whether released or not
# 
# **tagline:** The tagline of the movie

# In[ ]:


movies.info()


# In[ ]:


movies.budget =pd.to_numeric(movies.budget, errors='coerce')


# In[ ]:


movies.describe()


# In[ ]:


# Exploring the languages of the movies
pd.unique(movies['original_language'])


# In[ ]:


movies = movies[['id', 'original_title', 'original_language','vote_average','vote_count','adult','budget','revenue','runtime','status']]
movies = movies.rename(columns={'id':'movieId'})


# In[ ]:


mean_budget = movies['budget'].mean(skipna=True)
print (mean_budget)


# In[ ]:


movies['budget']=movies.budget.mask(movies.budget == 0,mean_budget)


# In[ ]:


mean_revenue = movies['revenue'].mean(skipna=True)
print (mean_revenue)


# In[ ]:


movies['revenue']=movies.revenue.mask(movies.revenue == 0,mean_revenue)


# In[ ]:


# Filtering English movie only
movies = movies[movies['original_language']== 'en'] 
movies.head()


# In[ ]:


movies.dtypes
ratings.dtypes
movies.movieId =pd.to_numeric(movies.movieId, errors='coerce')
ratings.movieId = pd.to_numeric(ratings.movieId, errors= 'coerce')


# In[ ]:


#creating a single dataframe merging the movie_data and ratings_data
df= pd.merge(ratings, movies, on='movieId', how='inner')
df.head()


# In[ ]:


df.info()


# ## Detecting null values and filling null values

# In[ ]:


df.isnull().sum() # or df.isna.sum()


# In[ ]:


df['status'].fillna(df['status'].mode()[0], inplace=True)


# In[ ]:


df['runtime'].fillna(df['runtime'].mode()[0], inplace=True)


# In[ ]:


df.isnull().sum()


# In[ ]:


df.describe()


# In[ ]:


ratings = pd.DataFrame(df.groupby('original_title')['rating'].mean().sort_values(ascending=False))
ratings.head(20)


# In[ ]:


ratings['number_of_ratings'] = df.groupby('original_title')['rating'].count()
ratings.head()


# In[ ]:


import matplotlib.pyplot as plt
#%matplotlib inline
ratings['rating'].hist(bins=50)
plt.title('Histogram');
plt.xlabel('Rating')
plt.ylabel('Number of movies')


# In[ ]:


ratings['number_of_ratings'].hist(bins=100)
plt.title('Histogram');
plt.xlabel('Number of ratings')
plt.ylabel('Number of movies')


# In[ ]:


import seaborn as sns
sns.jointplot(x='rating', y='number_of_ratings', data=ratings)


# ---
# ## Dataset Preparation
# 
# Now we will prepare the data for building our model for movie rating prediction

# ---
# ## Training Loop
# 
# Now we're ready to start the training process. First of all, let's split the original dataset using [train_test_split](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) function from the `scikit-learn` library. 

# In[ ]:


features=list(set(df.columns)-set(['rating'])-set(['original_language'])-set(['original_title'])) 
features

X1=df[features]
X= pd.get_dummies(X1)
X.astype(np.float64)
y=df['rating'].astype(np.int64)
X.head()


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4)
y_train.shape


# In[ ]:


# Data scaling
scaler = StandardScaler()

# Fit on training set only.
scaler.fit(X_train)

# Apply transform to both the training set and the test set.
X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)


# In[ ]:


# Random Forest

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, y_train)
acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)
acc_random_forest


# In[ ]:


acc_random_forest_test = round(random_forest.score(X_test, y_test) * 100, 2)
acc_random_forest_test


# In[ ]:


Y_pred


# In[ ]:


def MovieRecommender(df, MovieName, No_of_recommendation):
    movie_matrix = df.pivot_table(index='userId', columns='original_title', values='rating').fillna(0)
    movie_matrix.head(10)
    movie_user_rating = movie_matrix[MovieName]
    similar_to_movie=movie_matrix.corrwith(movie_user_rating)
    corr = pd.DataFrame(similar_to_movie, columns=['Correlation'])
    corr.dropna(inplace=True)
    corr = corr.join(ratings['number_of_ratings'])
    c=corr[corr['number_of_ratings'] > 50].sort_values(by='Correlation', ascending=False).head(No_of_recommendation)
    print(c)
    return c


# In[ ]:


c=MovieRecommender(df, MovieName='The Million Dollar Hotel', No_of_recommendation=5)


# # End of case study
