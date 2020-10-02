#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv("../input/tmdb_5000_credits.csv")
df.sample()


# In[ ]:


df1 = pd.read_csv("../input/tmdb_5000_movies.csv")
df1.sample()


# In[ ]:


df1 = df1.rename(columns={'id': 'movie_id'})
df_initial = pd.merge(df, df1, on=['movie_id','title'])
df_initial.sample()#.release_date.head().apply(lambda(x):x[:-2])


# In[ ]:


#Columns that contain `null` values
for i in list(df_initial.columns):
    if df_initial[i].isnull().values.any():
        print(i, end=",")


# In[ ]:


# replacing NaN values with unkowns and defaults
df_initial["homepage"].fillna("Unknown", inplace=True)
df_initial["overview"].fillna("Unknown", inplace=True)
df_initial["release_date"].fillna("2000-01-01", inplace=True)
df_initial["runtime"].fillna("0", inplace=True)
df_initial["tagline"].fillna("0", inplace=True)
df_initial.isnull().values.any()


# In[ ]:


df_Genre = pd.DataFrame(columns = ['movie_id','genre','revenue','title','vote_average'])
def pop_name(record):
    global df_Genre
    d = {}
    d['movie_id']=record['movie_id']
    genre_names = np.array([g['name'] for g in eval(record['genres'])])
    d['genre'] = []
    d['genre'].extend(genre_names)
    d['revenue']=record['revenue']
    d['title']=record['title']
    d['vote_average']=record['vote_average']
    df_Genre = df_Genre.append(pd.DataFrame(d), ignore_index=True, sort=True)
    
    
df_initial.apply(pop_name, axis=1)
df_Genre = df_Genre[['movie_id','genre','revenue','title','vote_average']]
df_Genre = df_Genre.infer_objects()
df_Genre.sample()


# Lets find out the genre in which maximum number of movies were made.

# In[ ]:


df_counts = pd.DataFrame(df_Genre['genre'].value_counts().reset_index())
df_counts.columns = ['Genre', 'Count']

sns.set(style="whitegrid")
ax = sns.barplot(x="Count", y="Genre", data=df_counts)

# f, ax = plt.subplots(figsize=(23, 9))
# sns.barplot(x = 'Count', y = 'Genre', data=df_counts.sort_values(by=['Genre', 'Count']))
# ax.set_title('.: occurences per genre :.')
# ax.set_xlabel('occurrences')
# ax.set_ylabel('genres')
# plt.show()


# The distribution is pretty clear, but lets try to create a wordcloud for the same

# In[ ]:


genre_frequency = df_counts.values.tolist()
genre_frequency[0:5]


# In[ ]:


from wordcloud import WordCloud, STOPWORDS
get_ipython().run_line_magic('matplotlib', 'inline')
words = dict()
token_frequency = genre_frequency
for s in token_frequency:
    words[s[0]] = s[1]
tone = 100 # define the color of the words
f, ax = plt.subplots(figsize=(14, 6))
wordcloud = WordCloud(width=550,height=300, background_color='black', 
                      max_words=1628,relative_scaling=0.6,
                      color_func=lambda *args, **kwargs: "cyan",
                      normalize_plurals=False)
wordcloud.generate_from_frequencies(words)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()


# Now, lets check out the genre that generated maximum revenue 

# In[ ]:


df_revenue = df_Genre.groupby('genre').agg('sum').reset_index()
sns.set(style="whitegrid")
ax = sns.barplot(x="revenue", y="genre", data=df_revenue)


# Let us restrict our search on the last 10 years

# In[ ]:


import datetime
decade_diff = int(datetime.datetime.now().strftime('%Y'))-10
df_initial['release_date'] = pd.to_datetime(df_initial['release_date'], 
                                            errors='coerce', format='%Y-%m-%d')

df_last_decade = df_initial.loc[df_initial['release_date'].dt.year > decade_diff]
df_last_decade.shape


# Let us check out the actors with more than 2 movies and an average vote > 5 for the last decade

# In[ ]:


df_Actor = pd.DataFrame(columns = ['actor','movie','release_year','vote_average'])
def pop_actor(record):
    global df_Actor
    d = {}
    actors = np.array([g['name'] for g in eval(record['cast'])])
    d['actor'] = []     
    d['actor'].extend(actors)
    d['movie'] = record['title']
    d['vote_average'] = record['vote_average']
    d['release_year'] = record['release_date'].year
    
    df_Actor = df_Actor.append(pd.DataFrame(d), ignore_index=True, sort=True)
    
df_last_decade[df_last_decade['vote_average']>5].apply(pop_actor, axis=1)
df_Actor = df_Actor[['actor','movie','release_year','vote_average']]
df_Actor = df_Actor.infer_objects()
df_Actor[df_Actor.vote_average>5].groupby('release_year')['actor'].value_counts().loc[lambda x:x>2]
#df_Actor.groupby([ 'release_year', 'actor']).size().unstack(fill_value=0).T


# Let us try to recommend top 5 movies based on each genre. 
# 
# We will consider movies with voting averge>5 and those which have generated great revenue

# In[ ]:


def recommend_by_genre(df, genre):
    list_genres = df['genre'].unique()
    movies = []
    #filter on movies that have votes higher than 5
    df_sample = df[(df['vote_average']>5) & (df['genre']==genre)]
    df_sample = df_sample.sort_values(by=['vote_average','revenue'], ascending=False).head()
    movies.extend(df_sample['title'])
    return movies

movies_by_genre = {}
for genre in df_Genre['genre'].unique():
    movies_by_genre[genre] = recommend_by_genre(df_Genre, genre)
    if movies_by_genre[genre] :
        print("Genre : " + genre )    
        print("Movies : " + '\n\t '.join(movies_by_genre[genre]))    

