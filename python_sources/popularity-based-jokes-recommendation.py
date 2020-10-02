#!/usr/bin/env python
# coding: utf-8

# ## What is Recommendation System ?
# 
# Recommender/recommendation system is a subclass of information filtering system that seeks to predict the rating/ preference a user would give to an item.
# 
# They are primarily used in applications where a person/ entity is involved with a product/ service. To further improve their experience with this product, we try to personalize it to their needs. For this we have to look up at their past interactions with this product.
# 
# *In one line* -> **Specialized content for everyone.**
# 
# *For further info, [Wiki](https://en.wikipedia.org/wiki/Recommender_system#:~:text=A%20recommender%20system%2C%20or%20a,would%20give%20to%20an%20item.)*
# 
# ## Types of Recommender System
# 
# * 1). Popularity Based
# * 2). Classification Based
# * 3). Content Based
# * 4). Collaborative Based
# * 5). Hybrid Based (Content + Collaborative)
# * 6). Association Based Rule Mining
# 
# ## Popularity based recommender system
# As the name suggests it recommends based on what is currently popular. This is particularly useful when you don't have past data as a reference to recommend product to the user. 
# 
# # Import packages and dataset

# In[ ]:


import pandas as pd
import numpy as np

data = pd.read_csv('../input/jester-online-joke-recommender/jesterfinal151cols.csv')
print(data.shape)
data.head()


# In[ ]:


#There are NaN values, we need to drop or impute them in data preprocessing step.
data.isnull().sum()


# # Data Preprocessing
# 
# Dataset contains no column headers. The first column is user id and subsequent columns are Joke ratings for 150 jokes. Also there are NaN values towards the end of the data
# 
# **Things to do:**
# * Add column headers
# * All other Joke rating columns would be renamed to 1-150
# * 0th column would be user_id
# * Some rows contain NaN values, replace them as 0
# * Many ratings are 99.0 such jokes were not rated by user, replace them as 0

# In[ ]:


#Convert all 151 columns into a range of 0-150
data.columns = range(data.shape[1]) #shape of column
print(data.columns) #Start 0, Stop 151, Step 1
data.head()


# In[ ]:


#0th column would be renamed to user_id
data.rename(columns = {0: 'user_id'}, inplace = True)
data.head()


# In[ ]:


#Replace all NaN values as 0
data = data.fillna(0)
data.tail()


# In[ ]:


#Replace all 99.0 ratings as 0
data = data.replace(99.0, 0)
data.head()


# Some these ratings are as high as 6.9 while some are -9.68. Lets normalize only ratings columns using **Standard Scalar**, the idea behind this is to transform your data such that it's distribution will have mean of 0 and standard deviation of 1. Standard scaler aligns it into a Gaussian or Normal disctribution.
# 
# *For further info on [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)*
# 
# **Things to do:**
# * Extract ratings
# * Fit Standard Scaler into Ratings

# In[ ]:


#Extract only ratings columns
ratings = data.iloc[:, 1:]
ratings


# In[ ]:


#Fit StandardScaler into ratings
from sklearn.preprocessing import StandardScaler
ratings_ss = StandardScaler().fit_transform(ratings)
ratings_ss


# # Recommend Popular Jokes
# Recommend the top n most popular jokes using mean ratings.
# 
# **Things to do:**
# * Find mean rating for all the jokes
# * Mean rating is an array that needs to be converted into Dataframe for sort into descending order
# * Recommend top n popular jokes

# In[ ]:


# Find the mean rating for all the jokes
mean_ratings = ratings_ss.mean(axis = 0) #axis of 0 for it to calculate mean across all rows 
print(mean_ratings.shape) #(150,) clearly indicates mean scores for all 150 jokes
mean_ratings


# In[ ]:


#Convert array into Dataframe and rename column name for better readability
mean_ratings = pd.DataFrame(mean_ratings)
mean_ratings.rename(columns = {0: 'mean_joke_ratings'}, inplace = True) 
mean_ratings


# In[ ]:


#Recommend the top n most popular jokes
n = 10
#mean_ratings.iloc[:,0].argsort()[:-(n+1):-1] #outputs only Joke ids
mean_ratings.sort_values(ascending = False, by = 'mean_joke_ratings')[:n] #outputs Joke ids and their mean ratings


# **Recommender system recommends the top 10 most popular jokes based on their mean ratings.**
