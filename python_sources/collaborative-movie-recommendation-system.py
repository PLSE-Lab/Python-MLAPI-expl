#!/usr/bin/env python
# coding: utf-8

# # A recommender system, or a recommendation system, is a subclass of information filtering system that seeks to predict the "rating" or "preference" a user would give to an item. They are primarily used in commercial applications. (source - Wikipedia)
# 
# ## Mainly three types of recommendation systems in machine learning based on filtering are used to suggest product and services to the consumers.
# 
# ### 1. Content Filtering
# 
# ### 2. Collaborative Filtering
# 
# ### 3. Hybrid Filtering
# 
# ## 1. Content Filtering:
# 
#    In this algorithm, we try finding items look alike. Once we have item look like matrix,
# 
#       we can easily recommend alike items to a customer, who has purchased any item from the store.
# 
# ## 2. Collaborative Filtering:
# 
#       Here, we try to search for look alike customers and offer products based on what his/her lookalike has chosen.
# 
#       This algorithm is very effective but takes a lot of time and resources.

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


tag=pd.read_csv('/kaggle/input/movielens-20m-dataset/tag.csv')
rating=pd.read_csv('/kaggle/input/movielens-20m-dataset/rating.csv')
movies=pd.read_csv('/kaggle/input/movielens-20m-dataset/movie.csv')
genome_scores=pd.read_csv('/kaggle/input/movielens-20m-dataset/genome_scores.csv')
link=pd.read_csv('/kaggle/input/movielens-20m-dataset/link.csv')
genome_tag=pd.read_csv('/kaggle/input/movielens-20m-dataset/genome_tags.csv')


# In[ ]:


movies.head()


# In[ ]:


link.head()


# In[ ]:


rating.head()


# In[ ]:


rating.shape


# In[ ]:


rating['userId'].value_counts().shape ## unique users


# In[ ]:


x=rating['userId'].value_counts()>500


# In[ ]:


y = x[x].index


# In[ ]:


y.shape


# In[ ]:


rating=rating[rating['userId'].isin(y)]


# In[ ]:


rating.shape


# In[ ]:


movie_details=movies.merge(rating,on='movieId')


# In[ ]:


movie_details.head()


# In[ ]:


movie_details.shape


# In[ ]:


movie_details.drop(columns=['timestamp'],inplace=True)


# In[ ]:


movie_details.shape


# In[ ]:


movie_details.head()


# In[ ]:


number_rating = movie_details.groupby('title')['rating'].count().reset_index()


# In[ ]:


number_rating.rename(columns={'rating':'number of rating'},inplace=True)


# In[ ]:


number_rating.head()


# In[ ]:


df=movie_details.merge(number_rating,on='title')


# In[ ]:


df.shape


# In[ ]:


df.head()


# In[ ]:


df=df[df['number of rating']>=50] #selecting valuable books by ratings


# In[ ]:


df.drop_duplicates(['title','userId'],inplace=True)


# In[ ]:


df.shape


# In[ ]:


df.head()


# In[ ]:


df.drop(columns=['number of rating'],inplace=True)


# In[ ]:


df.head()


# In[ ]:


df['rating']=df['rating'].astype(int)


# In[ ]:


df.head()


# In[ ]:


movie_pivot=df.pivot_table(columns='userId',index='title',values='rating')


# In[ ]:


movie_pivot.shape


# In[ ]:


movie_pivot.fillna(0,inplace=True)


# In[ ]:


movie_pivot


# In[ ]:


from scipy.sparse import csr_matrix
movie_sparse=csr_matrix(movie_pivot)


# In[ ]:




from sklearn.neighbors import NearestNeighbors
model=NearestNeighbors( n_neighbors=7,algorithm='brute',metric='cosine')


# In[ ]:


model.fit(movie_sparse)


# In[ ]:


df.drop(columns=['genres','userId','rating'],inplace=True)


# In[ ]:


df.drop_duplicates(inplace=True)


# In[ ]:


df.to_csv('codf.csv',index=False)


# In[ ]:


distances,suggestions=model.kneighbors(movie_pivot.iloc[540,:].values.reshape(1,-1))


# In[ ]:


distances


# In[ ]:


suggestions


# In[ ]:


df1=df.copy()
ti=[]
for i in df1['title']:
    ti.append(i.split(' (')[0])
df1['title']=ti


# In[ ]:




for i in range(len(suggestions)):
    print(movie_pivot.index[suggestions[i]])


# In[ ]:


def reco(movie_name):
    movie_id=df1[df1['title']=='Toy Story'].drop_duplicates('title')['movieId'].values[0]
    distances,suggestions=model.kneighbors(movie_pivot.iloc[movie_id,:].values.reshape(1,-1))
    
    
    
    for i in range(len(suggestions)):
        return (movie_pivot.index[suggestions[i]])


# In[ ]:


res=reco("It Conquered the World")


# In[ ]:


for i in res:
    print(i)

