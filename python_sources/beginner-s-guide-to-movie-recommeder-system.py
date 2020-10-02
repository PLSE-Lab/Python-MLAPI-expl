#!/usr/bin/env python
# coding: utf-8

# # Movie Recomender System

# ![](https://www.flashrouters.com/wp/wp-content/uploads/2015/01/ORSh4FD1.jpg)

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from ast import literal_eval

import warnings
warnings.filterwarnings('ignore')

import os
print(os.listdir("../input/tmdb-movie-metadata/"))


# In[ ]:


movies=pd.read_csv('../input/tmdb-movie-metadata/tmdb_5000_movies.csv')
credits=pd.read_csv('../input/tmdb-movie-metadata/tmdb_5000_credits.csv')


# In[ ]:


credits.head(1)


# In[ ]:


movies.head(1)


# In[ ]:


credits.columns=['id','title','cast','crew']


# In[ ]:


movies=movies.merge(credits,on='id')


# In[ ]:


movies.shape


# In[ ]:


movies.info()


# In[ ]:


movies.describe()


# In[ ]:


movies['genres']=movies['genres'].apply(literal_eval).apply(lambda x: [i['name'] for i in x])


# In[ ]:


movies['title']=movies['title_x']


# In[ ]:


movies.drop(['title_x','title_y'],axis=1,inplace=True)


# # Demographic Filtering
# 

# In[ ]:


vote_counts = movies[movies['vote_count'].notnull()]['vote_count'].astype('int')
vote_averages = movies[movies['vote_average'].notnull()]['vote_average'].astype('int')
C= vote_averages.mean()
C


# In[ ]:


m = vote_counts.quantile(0.95)
m


# In[ ]:


movies['year']=movies['release_date'].apply(lambda x: str(x).split('-')[0]  if x != np.nan else np.nan)


# In[ ]:


movies.head(1)


# In[ ]:


qualified=movies[(movies['vote_count']>=m) & (movies['vote_count'].notnull()) & (movies['vote_average'].notnull())][['title','year','vote_count','vote_average','popularity','genres']]


# In[ ]:


qualified.shape


# In[ ]:


def weighted_ratio(x):
    v=x['vote_count']
    R=x['vote_average']
    return (((v/(v+m))*R) + ((m/(v+m))*C))


# In[ ]:


qualified['wr']=qualified.apply(weighted_ratio,axis=1)


# In[ ]:


qualified['wr']=np.round(qualified['wr'],2)


# In[ ]:


qualified=qualified.sort_values(by='wr',ascending=False)


# In[ ]:


gen=movies.apply(lambda x: pd.Series(x['genres']),axis=1).stack().reset_index(level=1, drop=True)
gen.name='genre'
gen_movies=movies.drop('genres',axis=1).join(gen)


# In[ ]:


gen_movies.head(3)


# In[ ]:


def build_chart(genre,percentile=0.95):
    df=gen_movies[gen_movies['genre'] == genre]
    vote_counts=df[df['vote_count'].notnull()]['vote_count'].astype(int)
    vote_averages=df[df['vote_average'].notnull()]['vote_average'].astype(int)
    c=vote_averages.mean()
    m=vote_counts.quantile(percentile)
    
    qualified=df[(df['vote_count']>=m) & df['vote_count'].notnull() & df['vote_average'].notnull()][['title','year','vote_count','vote_average','popularity']]
    

    qualified['wr'] = qualified.apply(lambda x: (x['vote_count']/(x['vote_count']+m) * x['vote_average']) + (m/(m+x['vote_count']) * C), axis=1)
    qualified = qualified.sort_values('wr', ascending=False).head(250)
    
    return qualified


# In[ ]:


build_chart('Romance').head(10)


# In[ ]:


build_chart('Action')


# # Content Based Recomender System

# In[ ]:


movies['overview'].head()


# In[ ]:


#Import TfIdfVectorizer from scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer

#Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
tfidf=TfidfVectorizer(stop_words='english')

#Replace NaN with an empty string
movies['overview']=movies['overview'].fillna('')

#Construct the required TF-IDF matrix by fitting and transforming the data 
tfidf_matrix=tfidf.fit_transform(movies['overview'])

#Output the shape of tfidf_matrix
tfidf_matrix.shape


# In[ ]:


from sklearn.metrics.pairwise import linear_kernel

cosine_sim=linear_kernel(tfidf_matrix,tfidf_matrix)


# In[ ]:


indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()


# In[ ]:


def content_based(title, cosine_sim=cosine_sim):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices]


# In[ ]:


content_based('Black Swan')


# In[ ]:


content_based('The Avengers')


# # Collaborative Filtering

# In[ ]:


from surprise import Reader, Dataset, SVD
reader = Reader()
ratings = pd.read_csv('../input/the-movies-dataset/ratings_small.csv')
ratings.head()


# In[ ]:


data=Dataset.load_from_df(ratings[['userId','movieId','rating']], reader)
data


# In[ ]:


svd=SVD()
trainset = data.build_full_trainset()
svd.fit(trainset)


# In[ ]:


ratings[ratings['userId'] == 1]


# In[ ]:


svd.predict(1,206,5)

