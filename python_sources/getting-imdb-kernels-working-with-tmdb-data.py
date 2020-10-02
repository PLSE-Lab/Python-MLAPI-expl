#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# This kernel provides tools for converting the TMDB dataset that data into a format that will work best with kernels built on the IMDB dataset, then wraps up with an example of using code analysis to plan the update of another kernel.

# In[ ]:


import json
import pandas as pd


# ### Loading the data
# The TMDb data includes several json fields, so loading the data requires an extra step beyond reading the csv files.

# In[ ]:


def load_tmdb_movies(path):
    df = pd.read_csv(path)
    df['release_date'] = pd.to_datetime(df['release_date']).apply(lambda x: x.date())
    json_columns = ['genres', 'keywords', 'production_countries', 'production_companies', 'spoken_languages']
    for column in json_columns:
        df[column] = df[column].apply(json.loads)
    return df


def load_tmdb_credits(path):
    df = pd.read_csv(path)
    json_columns = ['cast', 'crew']
    for column in json_columns:
        df[column] = df[column].apply(json.loads)
    return df


# ### Functions for converting to the old format
# To convert the TMDb data to be as close as possible to the old format , copy the entire cell below into your kernel and run `convert_to_original_format`. Please note that the conversion isn't completely 1:1.

# In[ ]:


# Columns that existed in the IMDB version of the dataset and are gone.
LOST_COLUMNS = [
    'actor_1_facebook_likes',
    'actor_2_facebook_likes',
    'actor_3_facebook_likes',
    'aspect_ratio',
    'cast_total_facebook_likes',
    'color',
    'content_rating',
    'director_facebook_likes',
    'facenumber_in_poster',
    'movie_facebook_likes',
    'movie_imdb_link',
    'num_critic_for_reviews',
    'num_user_for_reviews'
                ]

# Columns in TMDb that had direct equivalents in the IMDB version. 
# These columns can be used with old kernels just by changing the names
TMDB_TO_IMDB_SIMPLE_EQUIVALENCIES = {
    'budget': 'budget',
    'genres': 'genres',
    'revenue': 'gross',
    'title': 'movie_title',
    'runtime': 'duration',
    'original_language': 'language',  # it's possible that spoken_languages would be a better match
    'keywords': 'plot_keywords',
    'vote_count': 'num_voted_users',
                                         }

IMDB_COLUMNS_TO_REMAP = {'imdb_score': 'vote_average'}


def safe_access(container, index_values):
    # return a missing value rather than an error upon indexing/key failure
    result = container
    try:
        for idx in index_values:
            result = result[idx]
        return result
    except IndexError or KeyError:
        return pd.np.nan


def get_director(crew_data):
    directors = [x['name'] for x in crew_data if x['job'] == 'Director']
    return safe_access(directors, [0])


def pipe_flatten_names(keywords):
    return '|'.join([x['name'] for x in keywords])


def convert_to_original_format(movies, credits):
    # Converts TMDb data to make it as compatible as possible with kernels built on the original version of the data.
    tmdb_movies = movies.copy()
    tmdb_movies.rename(columns=TMDB_TO_IMDB_SIMPLE_EQUIVALENCIES, inplace=True)
    tmdb_movies['title_year'] = pd.to_datetime(tmdb_movies['release_date']).apply(lambda x: x.year)
    # I'm assuming that the first production country is equivalent, but have not been able to validate this
    tmdb_movies['country'] = tmdb_movies['production_countries'].apply(lambda x: safe_access(x, [0, 'name']))
    tmdb_movies['language'] = tmdb_movies['spoken_languages'].apply(lambda x: safe_access(x, [0, 'name']))
    tmdb_movies['director_name'] = credits['crew'].apply(get_director)
    tmdb_movies['actor_1_name'] = credits['cast'].apply(lambda x: safe_access(x, [0, 'name']))
    tmdb_movies['actor_2_name'] = credits['cast'].apply(lambda x: safe_access(x, [1, 'name']))
    tmdb_movies['actor_3_name'] = credits['cast'].apply(lambda x: safe_access(x, [2, 'name']))
    tmdb_movies['genres'] = tmdb_movies['genres'].apply(pipe_flatten_names)
    tmdb_movies['plot_keywords'] = tmdb_movies['plot_keywords'].apply(pipe_flatten_names)
    return tmdb_movies


# Pulling it all together, we can now load a version of the TMDb data that has a very similar format to the original.

# In[ ]:


movies = load_tmdb_movies("../input/tmdb-movie-metadata/tmdb_5000_movies.csv")
credits = load_tmdb_credits("../input/tmdb-movie-metadata/tmdb_5000_credits.csv")
original_format =convert_to_original_format(movies, credits)


# ### A worked example
# I've taken [FabienDaniel's](https://www.kaggle.com/fabiendaniel) popular recommendation [Film Recommendation Engine](https://www.kaggle.com/fabiendaniel/film-recommendation-engine) kernel and gotten it running on the TMDb data. You can find [the converted version here](https://www.kaggle.com/sohier/film-recommendation-engine-converted-to-use-tmdb/).
# 
# The next section will show you how to scan your kernel to check for the number and locations of lines of code that reference the lost columns. We'll learn a bit about the inner workings of notebook structure along the way. Honestly though, it can be just as easy to run your kernel and see where it fails. The main advantage of the code analysis is that it provides a rough estimate of how much work it will be to perform the conversion.
# 
# I don't actually know anything about the .ipynb format, so let's see what a notebook looks like if we just read it as raw text. If the code is stored in a binary format or similar we may be out of luck.

# In[ ]:


with open("../input/static-copy-of-recommendation-engine-notebook/recommendation_engine.ipynb", "r") as f_open:
    raw_notebook = f_open.readlines()
for line in raw_notebook[:10]:
    print(line)


# Looks like json! This will be a huge help for stripping out the sections that we don't need.
# 
# I'm using a copy of the notebook stored as a dataset to ensure a consistent version is available over time, but you can access your own kernel's code from within that kernel with the following code:
# 
# ```with open("./__notebook_source__.ipynb", "r") as f_open:
#     raw_notebook = f_open.readlines()```
# 
# Now let's load the code in a dataframe so it's a bit easier to work with.

# In[ ]:


with open("../input/static-copy-of-recommendation-engine-notebook/recommendation_engine.ipynb", "r") as f_open:
    df = pd.DataFrame(json.load(f_open)['cells'])


# In[ ]:


df.head(3)


# In[ ]:


import re


def rows_with_lost_columns(code_lines): 
    lost_column_pattern = '\W(' + '|'.join(LOST_COLUMNS) + ')\W'
    # adding one  to the output since text editors usually use one based indexing.
    troubled_lines = [line_number + 1 for line_number, line in enumerate(code_lines)
                      if bool(re.search(lost_column_pattern, line))]
    if troubled_lines:
        return troubled_lines
    return None


# In[ ]:


df['lost_column_lines'] = df.apply(lambda row: rows_with_lost_columns(row['source'])
                                   if row['cell_type'] == 'code' else pd.np.nan, axis=1)


# Now we've got the cell number and line numbers of lines of code that will need to be dropped or changed.

# In[ ]:


num_lines_to_review = df.lost_column_lines.apply(lambda x: len(x) if type(x) == list else 0).sum()
print(f'{num_lines_to_review} lines of code need to be reviewed')


# Note that some of these are false positives; mostly from 'color' showing up in plotting calls.

# In[ ]:


df[['source', 'lost_column_lines']][~df.lost_column_lines.isnull()]


# Note that this method won't find all of the issues; it's merely a good way of estimating the scope of the required changes. 
# 
# Now that we've identified areas of concern, it's easiest to head over to [the converted kernel](https://www.kaggle.com/sohier/film-recommendation-engine-converted-to-use-tmdb/). I've noted where I've made changes so you can follow along there.
# 
# You'll see that many of the issues in converting the original [Film Recommendation Engine](https://www.kaggle.com/fabiendaniel/film-recommendation-engine) turned out to be related to the existence of json fields that aren't even directly referenced. For example, they break the a dataframe wide df.duplicate() method as lists/dictionaries aren't hashable.

# In[ ]:




