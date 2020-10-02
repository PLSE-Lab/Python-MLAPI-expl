#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


movies=pd.read_csv("/kaggle/input/newdataset/movies.csv")
ratings=pd.read_csv("/kaggle/input/newdataset/ratings.csv")


# In[ ]:


movies.head()


# In[ ]:


ratings.head()


# # Merging both csv files using movie ID

# In[ ]:


df=pd.merge(ratings,movies, on='movieId')


# In[ ]:


df.head(10)


# In[ ]:


df.shape


# In[ ]:


Ratings_avg=pd.DataFrame(df.groupby('title')['rating'].mean())
Ratings_avg.head()


# In[ ]:


#to print count of ratings


Ratings_avg['Rating count']=pd.DataFrame(df.groupby('title')['rating'].count())
Ratings_avg.head()


# In[ ]:


#Most rated movies of all time:

df.groupby('title')['rating'].count().sort_values(ascending=False).head()


# In[ ]:


userRatings=df.pivot_table(index='userId',columns='title',values='rating')
userRatings.head()


# In[ ]:


#Input user's Favourite movie to show the correlation calculation
movieName=input()


# In[ ]:


#Correlation Calculator

correlations=userRatings.corrwith(userRatings[movieName])
correlations


# In[ ]:


recommendation = pd.DataFrame(correlations,columns=['Correlation'])
recommendation.dropna(inplace=True)
recommendation = recommendation.join(Ratings_avg['Rating count'])
recommendation.head()


# In[ ]:


rec = recommendation[recommendation['Rating count']>100].sort_values('Correlation',ascending=False).reset_index()


# In[ ]:


rec.head()


# In[ ]:


rec = rec.merge(movies,on='title', how='left')
rec.head()


# ### Here is the list of movies that resemble the correlation with the movie input by the user.
