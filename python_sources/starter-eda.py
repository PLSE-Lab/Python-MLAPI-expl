#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")
plt.style.use("seaborn-pastel")

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # df_movies table
# 
# The table consists some of the important information about the movie titles. The table is the merged and trimmed version of the following IMDb original tables:
# * **title_basics** (tconst a.k.a. ID_title, primaryTitle, originalTitle, startYear, runtimeMinutes, genres, averageRating, numVotes,)
# * **title_principals** (nconst a.k.a. ID_crew, category, job, characters)
# * **title_crew** (director, writer)
# 
# The table is sorted by ID_title, missing titles are the ones which are not either movie or tv_movie. 
# * **Each title have it's number of crew times rows**
# * Also note that the above number multiplies by number of directors and writers (due to merging tables, therefore groupby function need to be used for single entries for each film)

# In[ ]:


df_movies = pd.read_csv("/kaggle/input/imdb-dataset-only-movie-ratings-and-crew/df_movies.csv")
df_movies.drop(['Unnamed: 0'],axis=1,inplace=True)
df_movies.head(10)


# # df_names table
# 
# The table is the trimmed version of "**names_basics**" table of IMDb's. It contains only the names which are involved in the movies.

# In[ ]:


df_names = pd.read_csv("/kaggle/input/imdb-dataset-only-movie-ratings-and-crew/df_names.csv")
df_names.set_index("nconst",drop=True,inplace=True)
df_names.head(10)


# ## Sample use of groupby for titles
# * By taking first rows from each group (movie title), you will miss information from the following columns:
#     * ID_crew, category, job and characters

# In[ ]:


grouped = df_movies.groupby("ID_title").first()
grouped.head(10)


# ## Top 10 directors with highest rating average
# Condition: Only directors having at least 5 films which are voted 25K or more are considered

# In[ ]:


top_dirs = grouped[grouped.numVotes>25000].groupby("director").agg(
                                              av_years=("startYear","median"),
                                              av_runtime=("runtimeMinutes","median"),
                                              av_rating=("averageRating","median"),
                                              av_vote_count=("numVotes","median"),
                                              no_of_films=("director","count")
                                              )

dirs_top10_by_rating = pd.merge(df_names["primaryName"],top_dirs[top_dirs.no_of_films>=5],left_index=True,right_index=True).sort_values("av_rating",ascending=False).head(10)

sns.barplot(data=dirs_top10_by_rating,x="av_rating",y="primaryName",orient="h")
for i,name in enumerate(dirs_top10_by_rating.primaryName):
    plt.text(dirs_top10_by_rating.av_rating[i],i,round(dirs_top10_by_rating.av_rating[i],1),va="center")
    plt.text(7.05,i,f"Av. Vote: {int(dirs_top10_by_rating.av_vote_count[i])}",va="center")
plt.xlim(7,8.5)
plt.show()


# ## Top 10 writers with highest rating average
# Condition: Only writers having at least 5 films which are voted 25K or more are considered

# In[ ]:


top_wris = grouped[grouped.numVotes>25000].groupby("writer").agg(
                                              av_years=("startYear","median"),
                                              av_runtime=("runtimeMinutes","median"),
                                              av_rating=("averageRating","median"),
                                              av_vote_count=("numVotes","median"),
                                              no_of_films=("writer","count")
                                              )

wris_top10_by_rating = pd.merge(df_names["primaryName"],top_wris[top_wris.no_of_films>=5],left_index=True,right_index=True).sort_values("av_rating",ascending=False).head(10)

sns.barplot(data=wris_top10_by_rating,x="av_rating",y="primaryName",orient="h")
for i,name in enumerate(wris_top10_by_rating.primaryName):
    plt.text(wris_top10_by_rating.av_rating[i],i,round(wris_top10_by_rating.av_rating[i],1),va="center")
    plt.text(7.05,i,f"Av. Vote: {int(wris_top10_by_rating.av_vote_count[i])}",va="center")
plt.xlim(7,8.5)
plt.show()


# 
