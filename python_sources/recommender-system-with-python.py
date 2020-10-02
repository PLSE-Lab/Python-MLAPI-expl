#!/usr/bin/env python
# coding: utf-8

# 
# # Recommender Systems with Python
# 
# Welcome to the code notebook for Recommender Systems with Python. In this lecture we will develop basic recommendation systems using Python and pandas. There is another notebook: *Advanced Recommender Systems with Python*. That notebook goes into more detail with the same data set.
# 
# In this notebook, we will focus on providing a basic recommendation system by suggesting items that are most similar to a particular item, in this case, movies. Keep in mind, this is not a true robust recommendation system, to describe it more accurately,it just tells you what movies/items are most similar to your movie choice.
# 
# There is no project for this topic, instead you have the option to work through the advanced lecture version of this notebook (totally optional!).
# 
# Let's get started!
# 
# ## Import Libraries

# In[ ]:


import numpy as np
import pandas as pd


# ## Get the Data

# In[ ]:


column_names = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('../input/u.csv', sep='\t', names=column_names)


# In[ ]:


df.head()


# Now let's get the movie titles:

# In[ ]:


movie_titles = pd.read_csv("../input/Movie_Id_Titles.csv")
movie_titles.head()


# We can merge them together:

# In[ ]:


df = pd.merge(df,movie_titles,on='item_id')
df.head()


# # EDA
# 
# Let's explore the data a bit and get a look at some of the best rated movies.
# 
# ## Visualization Imports

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
get_ipython().run_line_magic('matplotlib', 'inline')


# Let's create a ratings dataframe with average rating and number of ratings:

# In[ ]:


df.groupby('title')['rating'].mean().sort_values(ascending=False).head()


# In[ ]:


df.groupby('title')['rating'].count().sort_values(ascending=False).head()


# In[ ]:


ratings = pd.DataFrame(df.groupby('title')['rating'].mean())
ratings.head()


# Now set the number of ratings column:

# In[ ]:


ratings['num of ratings'] = pd.DataFrame(df.groupby('title')['rating'].count())
ratings.head()


# Now a few histograms:

# In[ ]:


plt.figure(figsize=(10,4))
ratings['num of ratings'].hist(bins=70)


# In[ ]:


plt.figure(figsize=(10,4))
ratings['rating'].hist(bins=70)


# In[ ]:


sns.jointplot(x='rating',y='num of ratings',data=ratings,alpha=0.5)


# Okay! Now that we have a general idea of what the data looks like, let's move on to creating a simple recommendation system:

# ## Recommending Similar Movies

# Now let's create a matrix that has the user ids on one access and the movie title on another axis. Each cell will then consist of the rating the user gave to that movie. Note there will be a lot of NaN values, because most people have not seen most of the movies.

# In[ ]:


moviemat = df.pivot_table(index='user_id',columns='title',values='rating')
moviemat.head()


# Most rated movie:

# In[ ]:


ratings.sort_values('num of ratings',ascending=False).head(10)


# Let's choose two movies: starwars, a sci-fi movie. And Toy Storyr, a Comedy/Animated.

# In[ ]:


ratings.head()


# Now let's grab the user ratings for those two movies:

# In[ ]:


starwars_user_ratings = moviemat['Star Wars (1977)']
toystory_user_ratings = moviemat['Toy Story (1995)']
starwars_user_ratings.head()


# We can then use corrwith() method to get correlations between two pandas series:

# In[ ]:


similar_to_starwars = moviemat.corrwith(starwars_user_ratings)
similar_to_toystory = moviemat.corrwith(toystory_user_ratings)


# Let's clean this by removing NaN values and using a DataFrame instead of a series:

# In[ ]:


corr_starwars = pd.DataFrame(similar_to_starwars,columns=['Correlation'])
corr_starwars.dropna(inplace=True)
corr_starwars.head()


# Now if we sort the dataframe by correlation, we should get the most similar movies, however note that we get some results that don't really make sense. This is because there are a lot of movies only watched once by users who also watched star wars (it was the most popular movie). 

# In[ ]:


corr_starwars.sort_values('Correlation',ascending=False).head(10)


# Let's fix this by filtering out movies that have less than 100 reviews (this value was chosen based off the histogram from earlier).

# In[ ]:


corr_starwars = corr_starwars.join(ratings['num of ratings'])
corr_starwars.head()


# Now sort the values and notice how the titles make a lot more sense:

# In[ ]:


corr_starwars[corr_starwars['num of ratings']>200].sort_values('Correlation',ascending=False).head()


# Now the same for the comedy/Animated Toy Story:

# In[ ]:


corr_toystory = pd.DataFrame(similar_to_toystory,columns=['Correlation'])
corr_toystory.dropna(inplace=True)
corr_toystory = corr_toystory.join(ratings['num of ratings'])
corr_toystory[corr_toystory['num of ratings']>200].sort_values('Correlation',ascending=False).head()

