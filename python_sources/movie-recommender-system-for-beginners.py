#!/usr/bin/env python
# coding: utf-8

# In this Kernel we will try to explore the movie data set and develop a movie recomender system.This Kernel is a work in process and if you like my work please do vote.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import matplotlib.pyplot as plt


# **Importing the movie Dataset** 

# In[ ]:


md = pd. read_csv('../input/the-movies-dataset/movies_metadata.csv')
md.head()


# We will be needing 'id' ,imdb_id,vote_average and title of the movie to carry out our work.So we can drop the unwanted colums

# In[ ]:


md.shape


# If the movie has less than 10 vote count we can ignore this from our analysis.As 10 people rating a movie may not give as a good idea about the movie

# In[ ]:


md = md[md['vote_count'] >= 50]  
md.head()


# In[ ]:


md.shape


# So the total number of customers in the list has come down to 22931

# In[ ]:



#md=md.iloc[:22000]


# In[ ]:


md.columns


# In[ ]:


md.drop(['adult','belongs_to_collection','budget','genres','homepage','original_language','original_title','overview','popularity','poster_path','production_companies','release_date','revenue','runtime','production_countries','spoken_languages','status','tagline','video'],axis=1, inplace=True)
md.head()


# In[ ]:


md.describe()


# So we can see that min vote = 0 , max vote =10 and average vote=5.61

# In[ ]:


#md.groupby('title').describe()


# We Dont want information on Id so we can ignore them while doing a groupby describe

# In[ ]:


md.groupby('title')['vote_average'].describe()


# In[ ]:


md1=md.drop(['id','imdb_id'],axis=1)
md1.head()


# In[ ]:


md1.shape


# Now our dataframe md1 has data on the name of the movie, it average rating and the number of times it was voted 

# **Plotting the Histograms**

# In[ ]:


md1['vote_average'].plot(bins=100,kind='hist',color='red')
plt.ioff()


# We can see from the distribution that most of the moves rating falls i the range 4 to 8

# In[ ]:


md1['vote_count'].plot(bins=80,kind='hist',color='red')
plt.xlim(xmin=0, xmax = 2000)
plt.ioff()


# From the graph we can make out that many moves have very less reviews.Arounf 150 movies have good number of reviews

# **Lets find out which moves have a rating of 10**

# In[ ]:


md1[md1['vote_average']==8]


# So the list contains 190 movies which have got a rating of 10.But we can note that very few people have given ratings for this movie

# **Lets see which movies have the highest numbers of reviews**

# In[ ]:


md1.sort_values('vote_count',ascending=False).head(10)


# **Lets see the movies with least number of votes**

# In[ ]:


md1.sort_values('vote_count',ascending=True).head(200)


# We can see that many movies dont have reviews or have very less reviews

# **Item Based Collabrative Filter **

# In[ ]:


md.info()


# In[ ]:


md.head()


# **Lets find out what movies are watched by user**

# In[ ]:


userid_movietitle_matrix=md.pivot_table(index='imdb_id',columns='title',values='vote_average')
userid_movietitle_matrix


# Now we have a matrix of movie matched by each user and the vote given by them

# In[ ]:


Inception=userid_movietitle_matrix['Inception']
Inception


# In[ ]:


Inception_correlations=pd.DataFrame(userid_movietitle_matrix.corrwith(Inception),columns=['Correlation'])
Inception_correlations


# In[ ]:





# In[ ]:




