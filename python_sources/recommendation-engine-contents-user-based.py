#!/usr/bin/env python
# coding: utf-8

# # Contents
# 1. [Data Importing & Exploring](#1)
# 2. [Data Preprocessing : making Cosine Similarities matrix](#2)
# 3. [Contents based Recommendation Engine](#3)
# 4. [User-based Collaborative Filtering Engine](#4)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity
from ast import literal_eval
import seaborn as sns
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# # <a id="1"></a><br> Data Importing & Exploring 

# In[ ]:


movies = pd.read_csv('../input/movies_metadata.csv', low_memory=False)
ratings = pd.read_csv('../input/ratings.csv')
credits = pd.read_csv('../input/credits.csv')
keywords = pd.read_csv('../input/keywords.csv')
links = pd.read_csv('../input/links.csv')


# In[ ]:


movies.head()


# **Histrgram for Genre distribution**

# In[ ]:


genre_frequencies = [y for x in movies.genres for y in x]


# In[ ]:


from collections import Counter
genre_count = dict(Counter(genre_frequencies))


# In[ ]:


plt.figure(figsize=(30,15))
sns.barplot(x=list(genre_count.keys()),y=list(genre_count.values()))
plt.title("Genre Type Histogram", fontsize=30)
plt.xticks(fontsize=15)
plt.xlabel("Genre Types",fontsize=25)
plt.yticks(fontsize=15)


# In[ ]:


plt.figure(figsize=(30,15))
sns.distplot(movies.runtime[movies.runtime<420],color='c')
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
plt.xlabel('Running Time',fontsize=25)
plt.ylabel('Frequencies',fontsize=25)
plt.title("Running Time histogram", fontsize=40)


# In[ ]:


plt.figure(figsize=(30,15))
sns.distplot(movies.runtime[movies.runtime<420],color='orange',bins=100)
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
plt.xlabel('Votes counts',fontsize=25)
plt.ylabel('Frequencies',fontsize=25)
plt.title("Vote counts Histogram", fontsize=40)


# In[ ]:


plt.figure(figsize=(30,15))
sns.distplot(list(movies.runtime[movies.runtime<420]),color='c',bins=100)
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
plt.xlabel('Votes Average Score',fontsize=25)
plt.ylabel('Frequencies',fontsize=25)
plt.title("Vote Average Score Histogram", fontsize=40)


# In[ ]:


ratings.userId.value_counts().head()


# # <a id="2"></a><br> Data Preprocessing : making Cosine Similarities matrix and Contents based Recommendation Engine

# In[ ]:


tfid = TfidfVectorizer(stop_words='english')
movies.overview = movies.overview.fillna('')

### since making cos_sim matrix taking too long, temporaily used only 10000 rows
tfid_matrix = tfid.fit_transform(movies.overview.iloc[1:10000])


# In[ ]:


tfid_matrix.shape


# https://www.google.com/url?sa=i&rct=j&q=&esrc=s&source=images&cd=&cad=rja&uact=8&ved=2ahUKEwjE-eeOkvLdAhUCZt4KHfRrCewQjRx6BAgBEAU&url=http%3A%2F%2Ftechinpink.com%2F2017%2F08%2F04%2Fimplementing-similarity-measures-cosine-similarity-versus-jaccard-similarity%2F&psig=AOvVaw1Jdc5prhjic09utk_5qqft&ust=1538926499074204

# In[ ]:


cos_sim = cosine_similarity(tfid_matrix,tfid_matrix)


# **Cosine Similarities Matrix**

# In[ ]:


pd.DataFrame(cos_sim).head(10)


# In[ ]:


indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()
indices.head()


# In[ ]:


idx = indices['Jumanji']
sim_movies = sorted(list(enumerate(cos_sim[idx])), key= lambda x: x[1], reverse=True)
sim_movies = sim_movies[1:11]
sim_num = [x[0] for x in sim_movies]
sim_value = [x[1] for x in sim_movies]
result = indices.iloc[sim_num]


# # <a id="3"></a><br> Contents based Recommendation Engine

# In[ ]:


### making recommend engine based on cosine similarities

def recommend_engine(title,cos_sim = cos_sim):
    idx = indices[title]
    sim_movies = sorted(list(enumerate(cos_sim[idx])), key= lambda x: x[1], reverse=True)
    sim_movies = sim_movies[1:11]
    sim_num = [x[0] for x in sim_movies]
    sim_value = [x[1] for x in sim_movies]
    result = indices.iloc[sim_num]
    result[0:10] = sim_value
    return(result)

### cosine sim value are represented with movie name


# In[ ]:


#### what is the recommended movies from Jumanji?
#### Most similar movies in terms of cosine similarities
recommend_engine('Jumanji')


# In[ ]:


movies.columns


# In[ ]:


credits.columns


# In[ ]:


### merge movies, keywords, credits data into movies sole dataset

movies = movies.drop([19730, 29503, 35587])

keywords['id'] = keywords['id'].astype('int')
credits['id'] = credits['id'].astype('int')
movies['id'] = movies['id'].astype('int')

movies = movies.merge(keywords, on='id')
movies = movies.merge(credits, on='id')


# In[ ]:


### strinfied features are splited into list

features = ['genres','keywords','cast','crew']

for feature in features:
    movies[feature] = movies[feature].apply(literal_eval)


# In[ ]:


### who is director? finding director function

def get_director(data):
    for x in data:
        if x['job'] == 'Director':
            return x['name']
    return np.nan


# In[ ]:


### making director columns

movies['director'] = movies.crew.apply(get_director)


# In[ ]:


### making get_list function
def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        if len(names) > 3:
            names = names[:3]
        return names

    return []


# In[ ]:


movies.cast = movies.cast.apply(get_list)
movies.genres = movies.genres.apply(get_list)
movies.keywords = movies.keywords.apply(get_list)


# In[ ]:


### delete space within strings and change into lowercase 
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''
        
### since our movies dataset is consiste of 3 columns of list and 1 string(director) column, 
### we divide function into two set.


# In[ ]:


features = ['cast','keywords','director','genres']


# In[ ]:


for feature in features:
    movies[feature] = movies[feature].apply(clean_data)


# In[ ]:


movies[features].head(10)


# In[ ]:


def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' '


# In[ ]:


movies['soup'] = movies.apply(create_soup, axis=1)


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(movies['soup'])


# In[ ]:


from sklearn.metrics.pairwise import cosine_similarity

cosine_sim2 = cosine_similarity(count_matrix, count_matrix)


# In[ ]:


movies = movies.reset_index()
indices = pd.Series(movies.index, index=movies['title'])


# In[ ]:


tmp = recommend_engine('The Godfather',cosine_sim2).index


# In[ ]:


list(tmp)


# In[ ]:


def rank_plot(movie_name, cos_sim=cos_sim):
    tmp = recommend_engine(movie_name,cos_sim)
    tmp2 = recommend_engine(movie_name,cosine_sim2)
    plt.figure(figsize=(10,5))
    sns.barplot(x = tmp[0:10], y=tmp.index)
    plt.title("Recommended Movies from  " + str.upper(movie_name) + " using cosine_sim", fontdict= {'fontsize' :20})
    plt.xlabel("Cosine Similarities")
    plt.show()
      
    plt.figure(figsize=(10,5))
    sns.barplot(x = tmp2[0:10], y=tmp2.index)
    plt.title("Recommended Movies from  " + str.upper(movie_name) + " using cosine_sim2", fontdict= {'fontsize' :20})
    plt.xlabel("Cosine Similarities")
    
    plt.show()


# In[ ]:


rank_plot("The Godfather")

rank_plot("Jumanji")


# # User-based Collaborative Filtering Engine

# In[ ]:


### importing rating dataset

rating = pd.read_csv("../input/ratings_small.csv")


# In[ ]:


### Checking data head

rating.head()


# In[ ]:


### Checking Data shape

rating.shape


# In[ ]:


### prepare Df which will record scores of movies

df = pd.DataFrame( index = rating.userId.unique() )


# In[ ]:


### Making df recording score of each user's record into df

for i in range(0,20000):
    ID = rating.loc[i,:].userId
    movieID = rating.loc[i,:].movieId
    movieScore = rating.loc[i,:].rating

    if movieID in list(df.columns):
        df.loc[ID, movieID] = movieScore
    else:
        df[movieID] = 0
        df.loc[ID,movieID] = movieScore


# In[ ]:


### shape of df (number of rows: number of users, number of columns : number of movies )

df.shape


# In[ ]:


### Checking data head

df.head()


# In[ ]:


### making cosine similarity matrix between users to users (671 by 671 in this case)

Filtering_cosim = cosine_similarity(df,df)


# In[ ]:


most_sim_user = sorted(list(enumerate(Filtering_cosim[100])), key=lambda x:x[1], reverse=True)[1]


# In[ ]:


most_sim_users = sorted(list(enumerate(Filtering_cosim[8])), key=lambda x: x[1], reverse=True)
most_sim_users = most_sim_users[1:11]
sim_users = [x[0] for x in most_sim_users]
print(sim_users)


# In[ ]:


candidates_movies = df.loc[sim_users,:]


# In[ ]:


def UBCF(user_num):
    ### finding most similar users among matrix

    most_sim_users = sorted(list(enumerate(Filtering_cosim[user_num])), key=lambda x: x[1], reverse=True)
    most_sim_users = most_sim_users[1:11]

    ### user index and their similairity values 

    sim_users = [x[0] for x in most_sim_users]
    sim_values = [x[1] for x in most_sim_users]

    ### among users having most similar preferences, finding movies having highest average score
    ### however except the movie that original user didn't see

    candidates_movies = df.loc[sim_users,:]

    candidates_movies.mean(axis=0).head()

    mean_score = pd.Series(candidates_movies.mean(axis=0))
    mean_score = mean_score.sort_values(axis=0, ascending=False)
    
    recom_mov = list(mean_score.iloc[0:10].keys())
    for i in recom_mov:
        recom_mov_title = movies.loc[movies.id.isin(recom_mov),:].title
        recom_mov_title
    return(recom_mov_title)


# # Which movies are recommended to each Users?
# 
# 

# In[ ]:


UBCF(400)


# In[ ]:


UBCF(1)


# # Things to be improved.....
#  I didn't composite cosine matrix from all of movies since huge data frame triggers RAM shortage in this kernel..
# 1.   User based Collaborative Engine is made so that recommend 10 movies for each users, but data memoery shorage blocking this in kernel.
# 2.  within same context,  cosine similarity matrix within a few users do not provide useful discrimination among users when it comes to efficiencies of engine.
