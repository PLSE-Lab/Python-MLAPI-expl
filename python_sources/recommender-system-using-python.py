#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# 
# Have you ever wondered how Netflix suggests movies to you based on the movies you have already watched? Or how does an e-commerce websites display options such as "Frequently Bought Together"? They may look relatively simple options but behind the scenes, a complex statistical algorithm executes in order to predict these recommendations. Such systems are called Recommender Systems, Recommendation Systems, or Recommendation Engines. A Recommender System is one of the most famous applications of data science and machine learning.
# 
# A Recommender System employs a statistical algorithm that seeks to predict users' ratings for a particular entity, based on the similarity between the entities or similarity between the users that previously rated those entities. The intuition is that similar types of users are likely to have similar ratings for a set of entities.
# 
# Currently, many of the big tech companies out there use a Recommender System in one way or another. You can find them anywhere from Amazon (product recommendations) to YouTube (video recommendations) to Facebook (friend recommendations). The ability to recommend relevant products or services to users can be a huge boost for a company, which is why it's so common to find this technique employed in so many sites.
# 
# In this Kernel, we will see how we can build a simple recommender system in Python.
# 
# 
# ## Types of Recommender Systems
# 
# There are two major approaches to build recommender systems: Content-Based Filtering and Collaborative Filtering:
# 
# ### Content-Based Filtering
# 
# In content-based filtering, the similarity between different products is calculated on the basis of the attributes of the products. For instance, in a content-based movie recommender system, the similarity between the movies is calculated on the basis of genres, the actors in the movie, the director of the movie, etc.
# 
# ### Collaborative Filtering
# 
# Collaborative filtering leverages the power of the crowd. The intuition behind collaborative filtering is that if a user A likes products X and Y, and if another user B likes product X, there is a fair bit of chance that he will like the product Y as well.
# 
# Take the example of a movie recommender system. Suppose a huge number of users have assigned the same ratings to movies X and Y. A new user comes who has assigned the same rating to movie X but hasn't watched movie Y yet. Collaborative filtering system will recommend him the movie Y.
# 
# 

# ### Movie Recommender System Implementation in Python
# 
# In this section, we'll develop a very simple movie recommender system in Python that uses the correlation between the ratings assigned to different movies, in order to find the similarity between the movies.
# 
# The dataset that we are going to use for this problem is the MovieLens Dataset. To download the dataset, go to https://grouplens.org/datasets/movielens/latest/ or "Movie Lens Small Latest Dataset" from kaggle datasets, download the "ml-latest-small.zip" file, which contains a subset of the actual movie dataset and contains 100000 ratings for 9000 movies by 700 users.
# 
# Once you unzip the downloaded file, you will see "links.csv", "movies.csv", "ratings.csv" and "tags.csv" files, along with the "README" document. In this article, we are going to use the "movies.csv" and "ratings.csv" files.

# ### Data Visualization and Preprocessing
# 
# The first step in every data science problem is to visualize and preprocess the data. We will do the same, so let's first import the "ratings.csv" file and see what it contains. Execute the following script:

# In[ ]:


import numpy as np
import pandas as pd

ratings_data = pd.read_csv('../input/movie-lens-small-latest-dataset/ratings.csv')
ratings_data.head()


# In the script above we use the read_csv() method of the Pandas library to read the "ratings.csv" file. Next, we call the head() method from the dataframe object returned by the read_csv() function, which will display the first five rows of the dataset.
# 
# You can see from the output that the "ratings.csv" file contains the userId, movieId, ratings, and timestamp attributes. Each row in the dataset corresponds to one rating. The userId column contains the ID of the user who left the rating. The movieId column contains the Id of the movie, the rating column contains the rating left by the user. Ratings can have values between 1 and 5. And finally, the timestamp refers to the time at which the user left the rating.
# 
# There is one problem with this dataset. It contains the IDs of the movies but not their titles. We'll need movie names for the movies we're recommending. The movie names are stored in the "movies.csv" file. Let's import the file and see the data it contains. Execute the following script:

# In[ ]:


movie_names = pd.read_csv('../input/movie-lens-small-latest-dataset/movies.csv')
movie_names.head()


# As you can see, this dataset contains movieId, the title of the movie, and its genre. We need a dataset that contains the userId, movie title, and its ratings. We have this information in two different dataframe objects: "ratings_data" and "movie_names". To get our desired information in a single dataframe, we can merge the two dataframes objects on the movieId column since it is common between the two dataframes.
# 
# We can do this using merge() function from the Pandas library, as shown below:

# In[ ]:


movie_data = pd.merge(ratings_data, movie_names, on='movieId')
movie_data.head()


# You can see our newly created dataframe contains userId, title, and rating of the movie as required.
# 
# Now let's take a look at the average rating of each movie. To do so, we can group the dataset by the title of the movie and then calculate the mean of the rating for each movie. We will then display the first five movies along with their average rating using the head() method. Look at the the following script:

# In[ ]:


movie_data.groupby('title')['rating'].mean().head()


# You can see that the average ratings are not sorted. Let's sort the ratings in the descending order of their average ratings:

# In[ ]:


movie_data.groupby('title')['rating'].mean().sort_values(ascending=False).head()


# The movies have now been sorted according to the ascending order of their ratings. However, there is a problem. A movie can make it to the top of the above list even if only a single user has given it five stars. Therefore, the above stats can be misleading. Normally, a movie which is really a good one gets a higher rating by a large number of users.
# 
# Let's now plot the total number of ratings for a movie:

# In[ ]:


movie_data.groupby('title')['rating'].count().sort_values(ascending=False).head()


# Now you can see some really good movies at the top. The above list supports our point that good movies normally receive higher ratings. Now we know that both the average rating per movie and the number of ratings per movie are important attributes. Let's create a new dataframe that contains both of these attributes.
# 
# Execute the following script to create ratings_mean_count dataframe and first add the average rating of each movie to this dataframe:

# In[ ]:


ratings_mean_count = pd.DataFrame(movie_data.groupby('title')['rating'].mean())


# In[ ]:


#Next, we need to add the number of ratings for a movie to the ratings_mean_count dataframe. Execute the following script to do so:
ratings_mean_count['rating_counts'] = pd.DataFrame(movie_data.groupby('title')['rating'].count())


# In[ ]:


ratings_mean_count.head()


# You can see movie title, along with the average rating and number of ratings for the movie.
# 
# Let's plot a histogram for the number of ratings represented by the "rating_counts" column in the above dataframe. Execute the following script:

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('dark')
get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize=(8,6))
plt.rcParams['patch.force_edgecolor'] = True
ratings_mean_count['rating_counts'].hist(bins=50)


# From the output, you can see that most of the movies have received less than 50 ratings. While the number of movies having more than 100 ratings is very low.
# 
# Now we'll plot a histogram for average ratings.

# In[ ]:


plt.figure(figsize=(8,6))
plt.rcParams['patch.force_edgecolor'] = True
ratings_mean_count['rating'].hist(bins=50)


# You can see that the integer values have taller bars than the floating values since most of the users assign rating as integer value i.e. 1, 2, 3, 4 or 5. Furthermore, it is evident that the data has a weak normal distribution with the mean of around 3.5. There are a few outliers in the data.
# 
# Earlier, we said that movies with a higher number of ratings usually have a high average rating as well since a good movie is normally well-known and a well-known movie is watched by a large number of people, and thus usually has a higher rating. Let's see if this is also the case with the movies in our dataset. We will plot average ratings against the number of ratings:
# 

# In[ ]:


plt.figure(figsize=(8,6))
plt.rcParams['patch.force_edgecolor'] = True
sns.jointplot(x='rating', y='rating_counts', data=ratings_mean_count, alpha=0.4)


# Finding Similarities Between Movies
# We spent quite a bit of time on visualizing and preprocessing our data. Now is the time to find the similarity between movies.
# 
# We will use the correlation between the ratings of a movie as the similarity metric. To find the correlation between the ratings of the movie, we need to create a matrix where each column is a movie name and each row contains the rating assigned by a specific user to that movie. Bear in mind that this matrix will have a lot of null values since every movie is not rated by every user.
# 
# To create the matrix of movie titles and corresponding user ratings, execute the following script:

# In[ ]:


user_movie_rating = movie_data.pivot_table(index='userId', columns='title', values='rating')
user_movie_rating.head()


# We know that each column contains all the user ratings for a particular movie. Let's find all the user ratings for the movie "Forrest Gump (1994)" and find the movies similar to it. We chose this movie since it has the highest number of ratings and we want to find the correlation between movies that have a higher number of ratings.
# 
# To find the user ratings for "Forrest Gump (1994)", execute the following script:

# In[ ]:


forrest_gump_ratings = user_movie_rating['Forrest Gump (1994)']
forrest_gump_ratings.head()


# Now let's retrieve all the movies that are similar to "Forrest Gump (1994)". We can find the correlation between the user ratings for the "Forest Gump (1994)" and all the other movies using corrwith() function as shown below:

# In[ ]:


movies_like_forest_gump = user_movie_rating.corrwith(forrest_gump_ratings)
corr_forrest_gump = pd.DataFrame(movies_like_forest_gump, columns=['Correlation'])
corr_forrest_gump.dropna(inplace=True)
corr_forrest_gump.head()


# In the above script, we first retrieved the list of all the movies related to "Forrest Gump (1994)" along with their correlation value, using corrwith() function. Next, we created a dataframe that contains movie title and correlation columns. We then removed all the NA values from the dataframe and displayed its first 5 rows using the head function.

# Let's sort the movies in descending order of correlation to see highly correlated movies at the top. Execute the following script:

# In[ ]:


corr_forrest_gump.sort_values('Correlation', ascending=False).head(10)


# From the output you can see that the movies that have high correlation with "Forrest Gump (1994)" are not very well known. This shows that correlation alone is not a good metric for similarity because there can be a user who watched '"Forest Gump (1994)" and only one other movie and rated both of them as 5.
# 
# A solution to this problem is to retrieve only those correlated movies that have at least more than 50 ratings. To do so, will add the rating_counts column from the rating_mean_count dataframe to our corr_forrest_gump dataframe. Execute the following script to do so:

# In[ ]:


corr_forrest_gump = corr_forrest_gump.join(ratings_mean_count['rating_counts'])
corr_forrest_gump.head()


# You can see that the movie 'burbs, The (1989)', which has the highest correlation has only 17 ratings. This means that only 17 users gave same ratings to "Forest Gump (1994)", "'burbs, The (1989)". However, we can deduce that a movie cannot be declared similar to the another movie based on just 17 ratings. This is why we added "rating_counts" column. Let's now filter movies correlated to "Forest Gump (1994)", that have more than 50 ratings. The following code will do that:
# 
# 

# In[ ]:


corr_forrest_gump[corr_forrest_gump ['rating_counts']>50].sort_values('Correlation', ascending=False).head()


# Now you can see from the output the movies that are highly correlated with "Forrest Gump (1994)". The movies in the list are some of the most famous movies Hollywood movies, and since "Forest Gump (1994)" is also a very famous movie, there is a high chance that these movies are correlated.
# 
# ## Conclusion
# 
# In this kernel, we studied what a recommender system is and how we can create it in Python using only the Pandas library. It is important to mention that the recommender system we created is very simple. Real-life recommender systems use very complex algorithms and will be discussed in a later.

# ### Reference
# 
# Inspired by "Usman Malik" from his blog https://stackabuse.com/creating-a-simple-recommender-system-in-python-using-pandas/
# 
# Learn more about recommender system methods https://towardsdatascience.com/how-to-build-from-scratch-a-content-based-movie-recommender-with-natural-language-processing-25ad400eb243
#  
