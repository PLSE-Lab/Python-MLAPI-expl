#!/usr/bin/env python
# coding: utf-8

# In this Kernel we will try to explore the movie data set and develop a movie recomender system.This Kernel is a work in process and if you like my work please do vote.

# In[ ]:


import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# **Importing the dataset**

# In[ ]:


movie_titles_df=pd.read_csv('../input/movie-titles/Movie_Id_Titles')
movie_titles_df.head()


# In[ ]:


movie_titles_df.shape


# We have 1682 movies in our dataset

# In[ ]:


column_names=['user_id','item_id','rating','timestamp']
movies_rating_df =pd.read_csv('../input/movielens/u.data.csv',sep='\t',names=column_names)

movies_rating_df.head()


# In[ ]:


movies_rating_df.shape


# We have once lakh customer reviews in the data set.Movies are rated on a scale of 1-5.Time stamp data is not useful for our analysis so we can drop it from our dataset

# In[ ]:


movies_rating_df.drop(['timestamp'],axis=1,inplace=True)


# In[ ]:


movies_rating_df.head()


# In[ ]:


movies_rating_df.describe()


# We can see that the minimum rating is 1 and maximum is 5 with an average of 3.52

# In[ ]:


movies_rating_df.info()


# There are no null values in the dataset.So no data cleanup is needed

# **Merging both the dataframes together**

# In[ ]:


movies_rating_df=pd.merge(movies_rating_df,movie_titles_df,on='item_id')


# In[ ]:


movies_rating_df.head()


# In[ ]:


movies_rating_df.shape


# **Vizualize the dataset**

# In[ ]:


movies_rating_df.groupby('title').describe()


# item_id and user_id can be removed to go the describe for the data

# In[ ]:


movies_rating_df.groupby('title')['rating'].describe()


# In[ ]:


rating_df_mean=movies_rating_df.groupby('title')['rating'].describe()['mean']


# In[ ]:


rating_df_mean.head()


# In[ ]:


rating_df_count=movies_rating_df.groupby('title')['rating'].describe()['count']


# In[ ]:


rating_df_count.head()


# Now we have the dataframes for mean and the count of the rating for the movies 

# **Lets Merge the mean and count data of rating**

# In[ ]:


rating_mean_count_df=pd.concat([rating_df_mean,rating_df_count],axis=1)


# In[ ]:


rating_mean_count_df.head()


# In[ ]:


rating_mean_count_df.reset_index().head()


# So now we have a dataframe with title,mean rating and number of times a movie has been rated.

# **Plotting distribution to get better idea**

# In[ ]:


rating_mean_count_df['mean'].plot(bins=100,kind='hist',color='r')
plt.ioff()


# It difficult to get 5 start ratings.The movies have higer rating may have few reviews.Lets check this out

# In[ ]:


rating_mean_count_df['count'].plot(bins=100,kind='hist',color='r')
plt.ioff()


# We can see clearly that the movies with high nummber of ratings are less.Many movies have less than 100 ratings

# In[ ]:


rating_mean_count_df[rating_mean_count_df['mean']==5]


# We can clearly see that movies with rating 5 have very few people rating it.

# **Getting movies with Mix reviews**

# In[ ]:


rating_mean_count_df.sort_values('count',ascending=False).head(10)


# So now we have the list of movies which have been rated most by people 

# **Getting the movies with least reviews **

# In[ ]:


rating_mean_count_df.sort_values('count',ascending=True).head(10)


# **Item Based Collabrative filtering **

# In[ ]:


movies_rating_df.head()


# **We will create a matrix using Pivot table**

# In[ ]:


userid_movietitle_matrix=movies_rating_df.pivot_table(index='user_id',columns='title',values='rating')
userid_movietitle_matrix


# This matrix gives us the the rating everu customer has given for the movie he or her has watched.

# **Getting the details for a particular movie**

# In[ ]:


titanic=userid_movietitle_matrix['Titanic (1997)']
titanic


# So we can see the ratings for titanic movie given by different users 

# **Creating correlations **

# In[ ]:


titanic_correlations=pd.DataFrame(userid_movietitle_matrix.corrwith(titanic),columns=['Correlations'])
titanic_correlations


# In[ ]:


titanic_correlations=titanic_correlations.join(rating_mean_count_df['count'])
titanic_correlations


# **Drop Nan**

# In[ ]:


titanic_correlations.dropna(inplace=True)
titanic_correlations


# In[ ]:


titanic_correlations.sort_values('Correlations',ascending=False).head(100)


# In[ ]:


titanic_correlations[titanic_correlations['count']>80].sort_values('Correlations',ascending=False).head(10)


# **Lets create item based collabarative filtering for whole dataset**

# In[ ]:


movie_correlations=userid_movietitle_matrix.corr(method='pearson',min_periods=80)
movie_correlations


# In[ ]:


myRatings = pd.read_csv("../input/my-movie-rating/My_Ratings.csv")
myRatings


# In[ ]:


myRatings['Movie Name'][0]


# In[ ]:


similar_movies_list=pd.Series()
for i in range(0,2):
    similar_movie=movie_correlations[myRatings['Movie Name'][i]].dropna()
    similar_movie=similar_movie.map(lambda x: x* myRatings['Ratings'][i])
    similar_movie_list=similar_movies_list.append(similar_movie)


# In[ ]:


similar_movie_list.sort_values(inplace=True,ascending=False)
similar_movie_list.head(10)

