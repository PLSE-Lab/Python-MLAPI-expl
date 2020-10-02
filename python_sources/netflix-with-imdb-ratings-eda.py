#!/usr/bin/env python
# coding: utf-8

# # Netflix titles and IMDb ratings - EDA

# In this notebook I am going to merge a netflix titles dataset from https://www.kaggle.com/shivamb/netflix-shows/kernels with IMDb dataset
# in order to get ratings values for the MOVIES available on netflix, and try to find some insights. 
# 
# 
# My exploratory data analysis of the netflix titles dataset can be foud here - https://www.kaggle.com/mykytazharov/eda-of-a-netflix-dataset-with-plotly-in-r .

# Note: the movies will be merged on title and the release year. Some films from the netflix dataset will not be found in the IMDb dataset. Further improvement of the merging can be implemented, as well as imputing the missing values.

# Import modules

# In[ ]:


import pandas as pd
import csv
import numpy
import seaborn as sns
import matplotlib.pyplot as plt


# Load IMDb data in .tsv format and save it into dataframes.
# I am going to use two datasets from the https://www.imdb.com/interfaces/. 
# Information courtesy of IMDb (http://www.imdb.com). Used with permission.
# The two datasets are: 
# * "title.basics.tsv.gz" - here I take a title of the movie and release year
# * "title.ratings.tsv.gz" - here I take ratings for the titles

# ## Data reading and cleaning

# In[ ]:


#read data into a datagrame
title_ratings=pd.read_csv("../input/movie-ratings-dataset/title.ratings.tsv/title.ratings.tsv", sep='\t')


# In[ ]:


title_ratings.head()


# In[ ]:


#number of rows in the dataframe
title_ratings.shape


# In[ ]:


#check if we have unique ratings for the titles
title_ratings.groupby(['tconst'], as_index=False).count()


# From the above output we see that we have unique rating values for each title.

# In[ ]:


title_basics=pd.read_csv("../input/movie-ratings-dataset/title.basics.tsv/title.basics.tsv", sep='\t')
title_basics=title_basics.drop_duplicates()


# In[ ]:


title_basics=title_basics[['titleType','tconst','primaryTitle', 'originalTitle', 'startYear']]
title_basics=title_basics[title_basics.titleType=='movie']
title_basics=title_basics[title_basics.startYear.apply(lambda x: str(x).isnumeric())]
title_basics.head()


# In[ ]:


title_basics.shape


# In[ ]:


grouped=title_basics.groupby(['primaryTitle', 'startYear'], as_index=False).count()
grouped.head()


# In[ ]:





# Now we join titles_basics and title_ratings dataframes on tconst (Index).

# In[ ]:


ratings_and_titles=pd.merge(title_ratings.set_index('tconst'), title_basics.set_index('tconst'), left_index=True, right_index=True, how='inner')
ratings_and_titles=ratings_and_titles.drop_duplicates()


# In[ ]:


ratings_and_titles.head()


# In[ ]:


ratings_and_titles.shape


# Now we load netflix titles dataset and join it with ratings_and_titles dataframe on title and primaryTitle.

# In[ ]:


netflix_titles=pd.read_csv("../input/netflix-shows/netflix_titles.csv", index_col="show_id")


# We drop rows where we dont have release_year.

# In[ ]:


netflix_titles=netflix_titles.dropna(subset=['release_year'])


# We need to change to integer the release_year column first.

# In[ ]:


netflix_titles.release_year=netflix_titles.release_year.astype(numpy.int64)


# Drop rows in ratings_and_titles with non-numeric values for startYear and convert to integer.

# In[ ]:


ratings_and_titles=ratings_and_titles[ratings_and_titles.startYear.apply(lambda x: str(x).isnumeric())]


# In[ ]:


ratings_and_titles.startYear=ratings_and_titles.startYear.astype(numpy.int64)


# Convert titles to lowercase.

# In[ ]:


netflix_titles['title']=netflix_titles['title'].str.lower()
ratings_and_titles['originalTitle']=ratings_and_titles['originalTitle'].str.lower()
ratings_and_titles['primaryTitle']=ratings_and_titles['primaryTitle'].str.lower()


# Now we can join netflix titles with IMDb ratings on title name and release year.

# In[ ]:


##subset movies
netflix_titles=netflix_titles[netflix_titles.type=='Movie']


# In[ ]:


netflix_titles.shape


# In[ ]:


netflix_titles_rating=pd.merge(netflix_titles, ratings_and_titles, left_on=['title', 'release_year'], right_on=['primaryTitle', 'startYear'], how='inner')


# ## Exploratory data analysis

# Sort the obtained data frame by averageRating and number of votes.

# In[ ]:


netflix_titles_rating.sort_values(by=['averageRating', 'numVotes'], inplace=True, ascending=False)


# In[ ]:


#look at titles where we have more than 2000 votes
netflix_titles_rating_2000=netflix_titles_rating[netflix_titles_rating.numVotes>2000]


# In[ ]:


netflix_titles_rating_2000.head(10)


# ### What is the distribution of the average ratings?

# In[ ]:


plt.figure(figsize=(20, 6))
sns.distplot(netflix_titles_rating['averageRating']);


# The output above shows something similar to the normal distribution.

# ### What is the distribution of number of votes?

# In[ ]:


plt.figure(figsize=(20, 6))
sns.distplot(netflix_titles_rating['numVotes']);


# ### What are the top ten movies on netflix according to the IMDb rating?

# In[ ]:


netflix_titles_rating_2000.head(10)['title']


# ### What are the countries producing the 100 most populat films?

# In[ ]:


plt.figure(figsize=(20, 6))
chart=sns.countplot(x="country", data=netflix_titles_rating_2000.head(100), order = netflix_titles_rating_2000.head(100)['country'].value_counts().index)
plt.xticks(
    rotation=45, 
    horizontalalignment='right',
    fontweight='light',
    fontsize='x-large'  
)


# ### What are the top genres of the 100 most popular films?

# Since movies may be listed in many genres, we need firstly to split them.

# In[ ]:


from itertools import chain

# return list from series of comma-separated strings
def chainer(s):
    return list(chain.from_iterable(s.str.split(',')))

# calculate lengths of splits
lens = netflix_titles_rating_2000.head(100)['listed_in'].str.split(',').map(len)

# create new dataframe, repeating or chaining as appropriate
res = pd.DataFrame({'title': numpy.repeat(netflix_titles_rating_2000.head(100)['title'], lens),
                    'listed_in': chainer(netflix_titles_rating_2000.head(100)['listed_in']),
                    })
res['listed_in']=res['listed_in'].str.strip()

print(res)


# In[ ]:


top_genres=res['listed_in'].value_counts()


# In[ ]:


top_genres


# In[ ]:


plt.figure(figsize=(20, 6))
chart=sns.countplot(x="listed_in", data=res, order = res['listed_in'].value_counts().index)
plt.xticks(
    rotation=45, 
    horizontalalignment='right',
    fontweight='light',
    fontsize='x-large'  
)


# In[ ]:


#save plot
chart.figure.savefig("pop_genres.png")


# In[ ]:


#save to the file
#netflix_titles_rating.to_csv('netflix_titles_rating_movies.csv')


# In[ ]:




