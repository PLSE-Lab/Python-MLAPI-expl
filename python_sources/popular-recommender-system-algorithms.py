#!/usr/bin/env python
# coding: utf-8

# # Popular Recommender System Algorithms
# 
# Recommendation systems are a collection of algorithms used to recommend items to users based on information taken from the user. These systems have become ubiquitous can be commonly seen in online stores, movies databases and job finders. In this kernel, we will explore different types of recommendation systems and implement them.

# In[1]:


import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
get_ipython().run_line_magic('matplotlib', 'inline')

import os
print(os.listdir("../input"))


# ## Data Preprocessing

# In[2]:


#Storing the movie information into a pandas dataframe
movies = pd.read_csv('../input/movie.csv')
movies.head()


# In[3]:


# Using regular expressions to find a year stored between parentheses
# We specify the parantheses so we don't conflict with movies that have years in their titles
movies['year'] = (movies.title.str.extract('(\(\d\d\d\d\))', expand=False)
                              .str.extract('(\d\d\d\d)', expand=False))  # Removing the parentheses

# Removing the years from the 'title' column
# Strip function to get rid of any ending whitespace characters that may have appeared
movies['title'] = (movies.title.str.replace('(\(\d\d\d\d\))', '')
                               .apply(lambda x: x.strip()))

# Every genre is separated by a | so we simply have to call the split function on |
movies['genres'] = movies.genres.str.split('|')
movies.head()


# In[4]:


movies.info() # Check for null elements


# Before we move on, let's compress our ratings size when reading to speed up to further processing. We can see(by looking at the data description) from userId column that the values are between 1 to 138.493. We can use int32 as its range (-2,147,483,648 to +2,147,483,647) contains what we need. For movieId, same thing also can be done. Lastly, we can convert rating column to float32.(Pandas doesn't support np.float16 for most of their operations so we have to stick with float32). Due to huge memory usage, we can further decrease our data by multiplying these columns with 2 to make everthing int and then convert back to np.int8.

# In[5]:


# Storing the user information into a pandas dataframe
ratings = pd.read_csv('../input/rating.csv', usecols=['userId', 'movieId', 'rating'],
                     dtype={'userId':np.int32, 'movieId':np.int32, 'rating':np.float32})
ratings.head()


# In[6]:


ratings['rating'] = ratings['rating'] * 2
ratings['rating'] = ratings['rating'].astype(np.int8)
ratings.info()


# In[7]:


ratings.head()


# ## Popularity Based Recommenders
# 
# In this part we are going to find the most popular movies and recommend them to users. This can be useful for newcomer users who don't know anything about the movies. 

# In[8]:


most_voted = (ratings.groupby('movieId')[['rating']]
                     .count()
                     .sort_values('rating', ascending=False)
                     .reset_index())
most_voted = pd.merge(most_voted, movies, on='movieId').drop('rating', axis=1)
most_voted.head()


# Our result shows that:
# - Pulp Fiction 
# - Forrest Gump
# - The Shawshank Redemption
# - The Silence of Lambs and 
# - Jurassic Park
# 
# are the most voted movies ever. So, based on our method we could suggest them to novices.

# ## Correlation Based on Recommenders (Item Based)
# 
# The next type of recommendation system to look at is correlation-based recommendation systems. These recommenders offer a basic form of collaborative filtering. That's because with correlation-based recommendation systems items are recommended based on similarities in their user review. In this sense, they do take user preferences into account. In these systems, you use Pearson's R correlation to recommend an item that is most similar to the item a user has already chosen. In other words, to recommend an item that has a review score that correlates with another item that a user has already chosen.

# In[9]:


# Due to problems with pandas, we can't use pivot_table with our all data as it throws MemoryError.
# Therefore, for this part we will work with a sample data
sample_ratings = ratings.sample(n=100000, random_state=20)

# Creating our sparse matrix and fill NA's with 0 to avoid high memory usage.
pivot = pd.pivot_table(sample_ratings, values='rating', index='userId', columns='movieId', fill_value=0)
pivot.head()


# In[11]:


pivot = pivot.astype(np.int8)
pivot.info()


# In[12]:


# Let's look something similar to Pulp Fiction
rand_movie = 296

similar = pivot.corrwith(pivot[rand_movie], drop=True).to_frame(name='PearsonR')


# In[13]:


rating_count = (ratings.groupby('movieId')[['rating']]
                       .count()
                       .sort_values('rating', ascending=False)
                       .reset_index())
rating_count = pd.merge(rating_count, movies, on='movieId')
rating_count.head()


# But let's think about this for a minute here. If we've found some movies that were really well correlated with Pulp Fiction but that had only, say, ten ratings total, then those movies probably wouldn't really be all that similar to Pulp Fiction. I mean maybe those movies got similar ratings, but they wouldn't be very popular. Therefore, that correlation really wouldn't be significant. We also need to take stock of how popular each of these movies is, in addition to how well the review scores correlate with the ratings that were given to other movies in the dataset. So to do that, we will join our corr data frame with a rating state of frame.

# In[14]:


similar_sum = similar.join(rating_count['rating'])
similar_top10 = similar_sum[similar_sum['rating']>=500].sort_values(['PearsonR', 'rating'], 
                                                            ascending=[False, False]).head(11)
# Add movie names
similar_top10 = pd.merge(similar_top10[1:11], movies[['title', 'movieId']], on='movieId')
similar_top10


# ## Model-based Collaborative Filtering Systems
# ## SVD Matrix Factorization
# 
# With these systems you build a model from user ratings, and then make recommendations based on that model. This offers a speed and scalability that's not available when you're forced to refer back to the entire dataset to make a prediction. We are going to see something called a utility matrix.
# 
# Utility matrix is also known as user item matrix. These matrices contain values for each user, each item, and the rating each user gave to each item. Another thing to note is that utility matrices are sparse because every user does not review every item. Actually, only a few users provide reviews for a few items. So in these matrices, we are likely to see mostly null values. Before explaining the truncated version, let's see the regular singular value decomposition or SVD.
# 
# SVD is a linear algebra method that you can use to decompose a utility matrix into three compressed matrices. It's useful for building a model-based recommender because we can use these compressed matrices to make recommendations without having to refer back to the complete and entire dataset. With SVD, we uncover latent variables. These are inferred variables that are present within and affect the behavior of a dataset. Although these variables are present and influential within a dataset, they're not directly observable. Now let's look at the anatomy of SVD.
# 
# Utility Matrix = U x S x V
# 
# We see three resultant matrices, U, S, and V. U is the left orthogonal matrix, and it holds the important,
# non-redundant information about users. On the right, we see matrix V. That's the right orthogonal matrix.
# It holds important, non-redundant information on items. In the middle, we see S, the diagonal matrix. This contains all of the information about the decomposition processes performed during the compression.
# 
# We want to use the similarities between users, to decide which movies to recommend, so we can use truncated SVD to compress all of the user ratings down to just small number of latent variables. These variables are going to capture most of the information that was stored in user columns previously. They represent a generalized view of users' tastes and preferences. The first thing we will do is to transpose our matrix, so that movies are represented by rows, and users are represented by columns. Then we'll use SVD to compress this matrix. All of the individual movie names will be retained along the rows. But the users will have been compressed down to number synthetic components which we will choose, that represent a generalized view of users' tastes.

# In[23]:


from sklearn.decomposition import TruncatedSVD

X = pivot.T
SVD = TruncatedSVD(n_components=500, random_state=20)
SVD_matrix = SVD.fit_transform(X)


# Let's see how much of these 500 variables cover the whole data

# In[24]:


SVD.explained_variance_ratio_.sum()


# We see that it covers about 52% of our whole data.
# 
# ### Generating a Correlation Matrix

# In[25]:


# We'll calculate the Pearson r correlation coefficient, 
# for every movie pair in the resultant matrix. With correlation being 
# based on similarities between user preferences.

corr_mat = np.corrcoef(SVD_matrix)
corr_mat.shape


# ### Isolating One Movie From the Correlation Matrix
# 
# Let's stick with Pulp Fiction choice

# In[45]:


corr_pulp_fiction = corr_mat[rand_movie]

# Recommending a Highly Correlated Movie.
# We will get different results due to decompression with svd
idx = X[(corr_pulp_fiction < 1.0) & (corr_pulp_fiction > 0.5)].index
movies.loc[idx+1, 'title']


# ## Conclusions
# 
# ### Advantages and Disadvantages of Content-Based Filtering
# 
# ##### Advantages
# * Learns user's preferences
# * Highly personalized for the user
# 
# ##### Disadvantages
# * Doesn't take into account what others think of the item, so low quality item recommendations might happen
# * Extracting data is not always intuitive
# * Determining what characteristics of the item the user dislikes or likes is not always obvious
# 
# ### Advantages and Disadvantages of Collaborative Filtering
# 
# ##### Advantages
# * Takes other user's ratings into consideration
# * Doesn't need to study or extract information from the recommended item
# * Adapts to the user's interests which might change over time
# 
# ##### Disadvantages
# * Approximation function can be slow
# * There might be a low of amount of users to approximate
# * Privacy issues when trying to learn the user's preferences
# 
# Don't forget to upvote my kernel if you like it :)
