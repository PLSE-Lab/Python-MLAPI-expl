#!/usr/bin/env python
# coding: utf-8

# # The Movies Dataset
# ###### Metadata on over 45,000 movies from TMDB (Movies database)
# 
# ## Import the metadata about movies : Merge, minor cleaning, export
# * Data source: The movies database, for ~45K movies in movielens data
# * DAta + kernel on kaggle datasets:  https://www.kaggle.com/rounakbanik/the-movies-dataset
# 
# * Additional dataset from TMDB (for 5K movies): https://www.kaggle.com/tmdb/tmdb-movie-metadata
# 
#     * relevant EDA kernels: 
#         * https://www.kaggle.com/rounakbanik/the-story-of-film
#         * https://www.kaggle.com/rounakbanik/movie-recommender-systems
#         
# ## Data description:
# * Context
# These files contain metadata for all 45,000 movies listed in the Full MovieLens Dataset. The dataset consists of movies released on or before July 2017. Data points include cast, crew, plot keywords, budget, revenue, posters, release dates, languages, production companies, countries, TMDB vote counts and vote averages.
# 
# 
# * Content
# This dataset consists of the following files:
# 
#     * movies_metadata.csv: The main Movies Metadata file. Contains information on 45,000 movies featured in the Full MovieLens dataset. Features include posters, backdrops, budget, revenue, release dates, languages, production countries and companies.
# 
#     * keywords.csv: Contains the movie plot keywords for our MovieLens movies. Available in the form of a stringified JSON Object.
# 
#     * credits.csv: Consists of Cast and Crew Information for all our movies. Available in the form of a stringified JSON Object.
# 
#     * links.csv: The file that contains the TMDB and IMDB IDs of all the movies featured in the Full MovieLens dataset.
# 
#     * The Full MovieLens Dataset consisting of 26 million ratings and 750,000 tag applications from 270,000 users on all the 45,000 movies in this dataset can be accessed here: https://grouplens.org/datasets/movielens/latest/
#     
#     * I don't attach here the user rating dataset
# 
# * Acknowledgements
# This dataset is an ensemble of data collected from TMDB and GroupLens. The Movie Details, Credits and Keywords have been collected from the TMDB Open API. This product uses the TMDb API but is not endorsed or certified by TMDb. Their API also provides access to data on many additional movies, actors and actresses, crew members, and TV shows. You can try it for yourself here: https://grouplens.org/datasets/movielens/latest/
# 
# The Movie Links and Ratings have been obtained from the Official GroupLens website. The files are a part of the dataset available here: https://grouplens.org/datasets/movielens/latest/
# 
# * Dan Ofer

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
# Any results you write to the current directory are saved as output.

from ast import literal_eval
import ast


# In[ ]:


PATH = "../input/"


# In[ ]:


df_credits = pd.read_csv(PATH+"credits.csv")
print("credits: \n ", df_credits.columns)
df_keywords = pd.read_csv(PATH+"keywords.csv")
print("\n keywords: \n", df_keywords.columns)

df_links = pd.read_csv(PATH+"links.csv")
print("\n links: \n", df_links.columns)


# In[ ]:


df = pd.read_csv(PATH+"movies_metadata.csv")
df.head()


# ### Some columns types are read badly : fix
# * ID column should be numeric.
# * we can see this using dtype (merge/join also would fail). 
# * it has no nulls though.
# * debug by printing non numeric rows in it:
#     * Finally: drop these (few) bad rows

# In[ ]:


df[["id","imdb_id"]].dtypes


# In[ ]:


df.isnull().sum()

# there's also 1 missing for cast/crew : we'll drop that row later


# In[ ]:


# Only 3 bad rows!
print (df[pd.to_numeric(df['id'], errors='coerce').isnull()])


# In[ ]:


df.shape


# In[ ]:


df["id"] =pd.to_numeric(df['id'], errors='coerce',downcast="integer")
# df["imdb_id"] =pd.to_numeric(df['imdb_id'], errors='coerce',downcast="integer")
df.dropna(subset=["id"],inplace=True)
df.shape


# In[ ]:


df.dropna(subset=["imdb_id"]).shape


# In[ ]:


# print(df.shape)
print("df_credits",df_credits.shape)


# In[ ]:


df_credits.head(3)


# In[ ]:


df.head(3)


# In[ ]:


df = df.merge(df_credits,on=["id"],how="left")


# In[ ]:


print(df_links.shape)
df_links.head()


# In[ ]:


df = df.merge(df_keywords,on=["id"],how="left")


# In[ ]:


print("missing imdb ids in main data:",df["imdb_id"].isnull().sum())
print("LINK data missing Imdb ids:",df_links["imdbId"].isnull().sum())
print("LINK data missing Tmdb ids:",df_links["tmdbId"].isnull().sum())
# looks like our coercion lost many rows. We don't really care about imdb IDs so we can ignore frankly..


# In[ ]:


# we see matching is poor for ID-movieId- but do we care? 
df.drop(["imdb_id"],axis=1).merge(df_links,left_on="id",right_on="movieId",how="inner").shape


# In[ ]:


# imdbID match rate: 
df.merge(df_links,left_on="imdb_id",right_on="imdbId",how="inner").shape


# ## We see we have poor matching with imdb/tmdb, likely related to data type
# * to avoid "down the road" confusion, so SKIPping merging with the IMDB, TMDB columns, as missing/partial/noisy data is worse than not having the columns!
#  * Can be fixed

# In[ ]:


# we don't merge due to bad ,atching - data wrangling issue, can be fixed
# df = df.merge(df_links,left_on="id",right_on="movieId",how="left")


# In[ ]:


df.shape


# In[ ]:


df.head()


# In[ ]:


df.shape


# In[ ]:


df.drop_duplicates().shape


# # some data cleaning: 
# * replace 0 budget with NaNs..
# * drop row with missing cast/crew
# 
# * 0 budget/revenue from: https://www.kaggle.com/rounakbanik/the-story-of-film

# In[ ]:


df.isnull().sum()


# In[ ]:


df.dropna(subset=["cast","crew","keywords","popularity"],inplace=True)
print(df.shape)


# In[ ]:


df['budget'] = pd.to_numeric(df['budget'], errors='coerce')
df['budget'] = df['budget'].replace(0, np.nan)

df['revenue'] = df['revenue'].replace(0, np.nan)


# ### Add form of weighted average (IMDB)
# * modified original formula as we don't care if a movie has 0 raings , so i add some noise/lambda 0
# Source:
# https://www.kaggle.com/rounakbanik/movie-recommender-systems
# 
# * Weighted Rating (WR) =  (vv+m.R)+(mv+m.C)(vv+m.R)+(mv+m.C) 
# where,
# 
# v is the number of votes for the movie
# m is the minimum votes required to be listed in the chart
# R is the average rating of the movie
# C is the mean vote across the whole report

# In[ ]:


vote_counts = df[df['vote_count'].notnull()]['vote_count'].astype('int')
vote_averages = df[df['vote_average'].notnull()]['vote_average'].astype('int')
C = vote_averages.mean()

m = vote_counts.quantile(0.75)

def weighted_rating(x):
    v = x['vote_count']+1 # added +1 - Dan
    R = x['vote_average']
    return (v/(v+m) * R) + (m/(m+v) * C)

df['weighted_rating'] = df.apply(weighted_rating, axis=1)


# ## clean list of genres/lists:
# * https://www.kaggle.com/rounakbanik/movie-recommender-systems
# 
# * lots "read list/dict as type then parse it into list, dropping the IDs.
# * I don't do this for cast data  -there's too much stuff there. 

# In[ ]:


df['genres'] = df['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])


# In[ ]:


df['cast'] = df['cast'].apply(literal_eval)
df['crew'] = df['crew'].apply(literal_eval)
df['keywords'] = df['keywords'].apply(literal_eval)
df['cast_size'] = df['cast'].apply(lambda x: len(x))
df['crew_size'] = df['crew'].apply(lambda x: len(x))


# In[ ]:


def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan
df['director'] = df['crew'].apply(get_director)


# ##### The literal_eval breaks  if nans in data - could be replaced with json load
# * original code works if nans removed.[](http://)
#     * https://www.kaggle.com/rounakbanik/movie-recommender-systems  = origin (used literal_eval)

# In[ ]:


# import json
# df['keywords'].iloc[0].replace("'",'"').str

# # works for 1 row:
# json.loads(df['keywords'].iloc[0].replace("'",'"'))


# In[ ]:


df['cast'] = df['cast'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
df['cast'] = df['cast'].apply(lambda x: x[:3] if len(x) >=3 else x)

df['keywords'] = df['keywords'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])


# In[ ]:


df['production_countries'].head()


# In[ ]:


df['production_countries'] = df['production_countries'].fillna('[]').apply(ast.literal_eval)
df['production_countries'] = df['production_countries'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])


# In[ ]:


df['production_countries'].head(2)


# In[ ]:


df['revenue_divide_budget'] = df['revenue'] / df['budget']


# * is in franchise?
# * https://www.kaggle.com/rounakbanik/the-story-of-film
#     * https://www.kaggleusercontent.com/kf/1657626/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..eEbV7U4MaFNpa7b0YfReIA.IN9Jh-3aT-tHObvb4Sb0OcwP7PuwPCUfLWa0_ylT6IHO3NEjlTmx7S-hWZF2Z1MJkM7nw6gBNYBi76sEElfDTNhwkDVg2obqCotynu9c7c-xkoRCDxkm7KGMiCkRW_7e.-ceLL39hoIgr4W6iO1nLMA/__results__.html#Franchise-Movies

# In[ ]:


df['belongs_to_collection'].head()


# ## I couldn't get the collections working with possible nans, skipping

# In[ ]:


# https://stackoverflow.com/questions/26614465/python-pandas-apply-function-if-a-column-value-is-not-null

# df['belongs_to_collection'] = df['belongs_to_collection'].apply(lambda row: ast.literal_eval(row) if (row.notnull().all()) , axis=1).apply(lambda x: x['name'] if isinstance(x, dict) else np.nan)

#  df['belongs_to_collection'].apply(lambda row: ast.literal_eval(row) if (row.notnull().all()) else row)

#  df['belongs_to_collection'].apply(lambda row: ast.literal_eval(row) if (row.isnull()== False) else row)


# In[ ]:


df['belongs_to_collection'] = df['belongs_to_collection'].fillna("[]").apply(ast.literal_eval).apply(lambda x: x['name'] if isinstance(x, dict) else np.nan)


# In[ ]:


## https://www.kaggle.com/rounakbanik/the-story-of-film
df['spoken_languages'] = df['spoken_languages'].fillna('[]').apply(ast.literal_eval).apply(lambda x: len(x) if isinstance(x, list) else np.nan)


# In[ ]:


df.head(20)['production_companies'].fillna("[]").apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])


# In[ ]:


df['production_companies'] = df['production_companies'].apply(ast.literal_eval)
df['production_companies'] = df['production_companies'].fillna("[]").apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])


# In[ ]:


df.head(3)


# In[ ]:


df.shape


# In[ ]:


df.to_csv("movies_tmdbMeta.csv.gz",index=False,compression="gzip")


# In[ ]:




