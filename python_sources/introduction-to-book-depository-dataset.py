#!/usr/bin/env python
# coding: utf-8

# # Book Depository Dataset EDA
# Through this notebook we will try to become familiar `Book Depository Dataset` and extract some usefull insights. The goal of this notebook is to become an introductory step for the dataset.

# In[ ]:


import pandas as pd
import os
import json
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Dataset Structure
# Files:
#  - `categories.csv`
#  - `dataset.csv`
#  - `formats.csv`
#  - `places.csv`
# 
# The dataset consists of 5 file, the main `dataset.csv` file and some extra files. Extra files works as lookup tables for category, author, format and publication place. The reason behind this decision was to prevent data redundancy.
# 
# Fields:
# 
#  * `authors`: Book's author(s) (`list of str`)
#  * `bestsellers-rank`: Bestsellers ranking (`int`)
#  * `categories`: Book's categories. Check `authors.csv` for mapping (`list of int`)
#  * `description`: Book description (`str`)
#  * `dimension_x`: Book's dimension X (`float` cm)
#  * `dimension_y`: Book's dimension Y (`float` cm)
#  * `dimension_z`: Book's dimension Z (`float` mm)
#  * `edition`: Edition (`str`)
#  * `edition-statement`: Edition statement (`str`)
#  * `for-ages`: Range of ages (`str`)
#  * `format`: Book's format. Check `formats.csv` for mapping (`int`)
#  * `id`: Book's unique id (`int`)
#  * `illustrations-note`: 
#  * `imprint`: 
#  * `index-date`: Book's crawling date (`date`)
#  * `isbn10`: Book's ISBN-10 (`str`)
#  * `isbn13`: Book's ISBN-13 (`str`)
#  * `lang`: List of book' language(s)
#  * `publication-date`: Publication date (`date`)
#  * `publication-place`: Publication place (`id`)
#  * `publisher`: Publisher (`str`)
#  * `rating-avg`: Rating average [0-5] (`float`)
#  * `rating-count`: Number of ratings
#  * `title`: Book's title (`str`)
#  * `url`: Book relative url (https://bookdepository.com + `url`)
#  * `weight`: Book's weight (`float` gr)
# 
# So, lets assign each file to a different dataframe

# In[ ]:


if os.path.exists('../input/book-depository-dataset'):
    path_prefix = '../input/book-depository-dataset/{}.csv'
else:
    path_prefix = '../export/kaggle/{}.csv'


df, df_f, df_a, df_c, df_p = [
    pd.read_csv(path_prefix.format(_)) for _ in ('dataset', 'formats', 'authors', 'categories', 'places')
]


# In[ ]:


# df = df.sample(n=500)
df.head()


# ## Basic Stats
# Firtly, lets display some basic statistics:

# In[ ]:


df.describe()


# **Publication Date Distribution**:
# Most books are published in t

# In[ ]:


df["publication-date"] = df["publication-date"].astype("datetime64")
df.groupby(df["publication-date"].dt.year).id.count().plot(title='Publication date distribution')


# In[ ]:


df["index-date"] = df["index-date"].astype("datetime64")
df.groupby(df["index-date"].dt.month).id.count().plot(title='Crawling date distribution')


# In[ ]:


df.groupby(['lang']).id.count().sort_values(ascending=False)[:5].plot(kind='pie', title="Most common languages")


# In[ ]:


import math
sns.lineplot(data=df.groupby(df['rating-avg'].dropna().apply(int)).id.count().reset_index(), x='rating-avg', y='id')


# In[ ]:


dims = pd.DataFrame({
    'dims': df['dimension-x'].fillna('0').astype(int).astype(str).str.cat(df['dimension-y'].fillna('0').astype(int).astype(str),sep=" x ").replace('0 x 0', 'Unknown').values, 
    'id': df['id'].values
})
dims.groupby(['dims']).id.count().sort_values(ascending=False)[:8].plot(kind='pie', title="Most common dimensions")


# In[ ]:


pd.merge(
    df[['id', 'publication-place']], df_p, left_on='publication-place', right_on='place_id'
).groupby(['place_name']).id.count().sort_values(ascending=False)[:8].plot(kind='pie', title="Most common publication places")

