#!/usr/bin/env python
# coding: utf-8

# ![TMDB](https://i.imgur.com/nCUVhIO.jpg)
# 
# # TMDB - Exploratory Data Analysis
# 
# Analysis with graph and very simple data extraction from the TMDB dataset. I've tried to be curious and find some interesting stuff inside this data. Feel free to comment or suggest some updates or other information that you found interesting!
# 
# # Remember to press the UP button!!
# 
# #### Goal:
# Using metadata on over 7,000 past films from The Movie Database and predict their overall worldwide box office revenue.
# 
# ## Reading data

# In[ ]:


import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

import re
from os import path, getcwd
from datetime import datetime

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error

from PIL import Image
from wordcloud import WordCloud, STOPWORDS

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from IPython.display import display, HTML

pd.set_option('display.max_columns', 100)
plt.style.use('bmh')

ID = 'id'
TARGET = 'revenue'
NFOLDS = 5
SEED = 126
NROWS = None
DATA_DIR = '../input'

TRAIN_FILE = f'{DATA_DIR}/train.csv'
TEST_FILE = f'{DATA_DIR}/test.csv'


# In[ ]:


train_data = pd.read_csv(TRAIN_FILE, nrows=NROWS)
test_data = pd.read_csv(TEST_FILE, nrows=NROWS)


# ## Extract basic information
# 
# Evaluate pythonic object into real data.

# In[ ]:


series_cols = ['belongs_to_collection', 'genres', 'production_companies',
               'production_countries', 'spoken_languages', 'Keywords',
               'cast', 'crew']
train = train_data.copy()
test = test_data.copy()
for df in [train, test]:
    for column in series_cols:
        df[column] = df[column].apply(lambda s: [] if pd.isnull(s) else eval(s))

full = pd.concat([train, test], sort=False)


# Count unique values for some features, and create a DataFrame from that.

# In[ ]:


def uniqueValues(df, col, key):
    all_values = []
    for record in df[col]:
        lst = [d[key] for d in record]
        all_values.extend(lst)
    all_values = np.array(all_values)
    unique, counts = np.unique(all_values, return_counts=True)
    return pd.DataFrame({ 'Value': unique, 'Counts': counts })

genres_unique = uniqueValues(full, 'genres', 'name').sort_values(by='Counts', ascending=False)
languages_unique = uniqueValues(full, 'spoken_languages', 'iso_639_1').sort_values(by='Counts', ascending=False)
top_languages = languages_unique.iloc[:4]

test.loc[test['release_date'].isnull() == True, 'release_date'] = '01/01/00'


# In[ ]:


def fixYear(row):
    year = int(row.split('/')[2])
    return row[:-2] + str(year + (2000 if year <= 19 else 1900))

def extractField(row, value):
    if row is np.nan: return 0
    return 1 if value in row else 0

for df in [train, test]:
    df['genres_list'] = df['genres'].apply(lambda row: ','.join(d['name'] for d in row))
    df['genres_count'] = df['genres'].apply(lambda x: len(x))

    df['budget_to_popularity'] = df['budget'] / df['popularity']
    df['budget_to_runtime'] = df['budget'] / df['runtime']

    df['prod_companies_list'] = df['production_companies'].apply(lambda row: ','.join(d['name'] for d in row))
    df['prod_countries_list'] = df['production_countries'].apply(lambda row: ','.join(d['iso_3166_1'] for d in row))

    df['languages_list'] = df['spoken_languages'].apply(lambda row: ','.join(d['iso_639_1'] for d in row))

    for l in top_languages['Value'].values:
        df['lang_' + l] = df['languages_list'].apply(extractField, args=(l,))

    df['has_homepage'] = df['homepage'].apply(lambda v: pd.isnull(v) == False)

    df['release_date'] = df['release_date'].apply(fixYear)
    df['release_date'] = pd.to_datetime(df['release_date'])

    date_parts = ['year', 'weekday', 'month', 'weekofyear', 'day', 'quarter']
    for part in date_parts:
        part_col = 'release_date' + '_' + part
        df[part_col] = getattr(df['release_date'].dt, part).astype(int)

    df['collection'] = df['belongs_to_collection'].apply(lambda row: ','.join(d['name'] for d in row))
    df['has_collection'] = df['collection'].apply(lambda v: 1 if v else 0)


# # Ready for analyze?
# 
# <img src="https://i.ytimg.com/vi/UXm2cg-fOKU/maxresdefault.jpg" alt="ironman" width="70%"/>

# In[ ]:


train.sample(2).T


# In[ ]:


full = pd.concat([train, test], sort=False)


# ## Revenue - I got some bad ideas in my head!
# 
# <img src="https://static.rogerebert.com/uploads/review/primary_image/reviews/great-movie-taxi-driver-1976/hero_Taxi-Driver-image.jpg" alt="taxidriver" width="70%"/>

# In[ ]:


fig = plt.figure(figsize = (20, 6))
plt.subplot(1, 2, 1)
sns.distplot(train['revenue'])
plt.subplot(1, 2, 2)
sns.distplot(np.log1p(train['revenue']))
fig.suptitle('Revenue', fontsize=20)
plt.show()


# In[ ]:


plt.figure(figsize=(8, 8))
plt.scatter(train['release_date_year'], train['revenue'])
plt.title('Movie revenue per year')
plt.xlabel('Year')
plt.ylabel('Revenue')
plt.show()


# ### Top Movies by revenue

# In[ ]:


top_movies = train.sort_values(by='revenue', ascending=False)
top_movies.head(10)[['title', 'revenue']]


# ## Profit
# In accounting, gross profit, gross margin, sales profit, or credit sales is the difference between revenue and the cost of making a product or providing a service. [Wikipedia](https://en.wikipedia.org/wiki/Gross_profit)

# In[ ]:


train['profit'] = train.apply(lambda row: row['revenue'] - row['budget'], axis=1)


# In[ ]:


plt.figure(figsize = (16, 6))
sns.distplot(train['profit'])
plt.title('Profit')
plt.show()


# In[ ]:


sns.lmplot('revenue', 'budget', data=train)
plt.show()


# ### Top Movies by Profit

# In[ ]:


worst_movies = train.sort_values(by='profit', ascending=False)
worst_movies.head(10)[['title', 'profit', 'budget', 'revenue']]


# ### Worst Movies by Profit

# In[ ]:


worst_movies = train.sort_values(by='profit', ascending=True)
worst_movies.head(10)[['title', 'profit', 'budget', 'revenue']]


# ## Year

# The number of movies is increasing year after year. The TMDB project was started in **2008**.

# In[ ]:


plt.figure(figsize=(8, 8))
dataTrain = train['release_date_year'].value_counts().sort_index()
dataTest = test['release_date_year'].value_counts().sort_index()
plt.plot(dataTrain.index, dataTrain.values, label='train')
plt.plot(dataTest.index, dataTest.values, label='test')
plt.title('Number of movies per year')
plt.xlabel('Year')
plt.ylabel('Revenue')
plt.legend(loc='upper center', frameon=False)
plt.show()


# ## Popularity

# In[ ]:


fig = plt.figure(figsize = (20, 6))
plt.subplot(1, 2, 1)
sns.distplot(full['popularity'])
plt.subplot(1, 2, 2)
sns.distplot(np.log1p(full['popularity']))
fig.suptitle('Popularity (full)', fontsize=20)
plt.show()


# In[ ]:


plt.figure(figsize=(20, 8))

plt.subplot(1, 2, 1)
plt.scatter(full['popularity'], full['revenue'])
plt.title('Popularity vs revenue (full)')
plt.xlabel('Popularity')
plt.ylabel('Revenue')

plt.subplot(1, 2, 2)
plt.scatter(np.log1p(full['popularity']), np.log1p(full['revenue']))
plt.title('Popularity vs revenue - log(x + 1) (full)')
plt.xlabel('Popularity')
plt.ylabel('Revenue')

plt.show()


# Is difficult to estimate populairty for older movies.

# In[ ]:


plt.figure(figsize=(8, 8))
plt.scatter(full['release_date_year'], full['popularity'])
plt.title('Popularity per year (full)')
plt.xlabel('Year')
plt.ylabel('Popularity')
plt.show()


# ## Budget

# In[ ]:


fig = plt.figure(figsize = (20, 6))
plt.subplot(1, 2, 1)
sns.distplot(full['budget'])
plt.subplot(1, 2, 2)
sns.distplot(np.log1p(full['budget']))
fig.suptitle('Budget (full)', fontsize=20)
plt.show()


# In[ ]:


plt.figure(figsize=(20, 8))

plt.subplot(1, 2, 1)
plt.scatter(train['budget'], train['revenue'])
plt.title('Budget vs revenue')
plt.xlabel('Budget')
plt.ylabel('Revenue')

plt.subplot(1, 2, 2)
plt.scatter(np.log1p(train['budget']), np.log1p(train['revenue']))
plt.title('Budget vs revenue - log(x + 1)')
plt.xlabel('Budget')
plt.ylabel('Revenue')

plt.show()


# ## Runtime

# In[ ]:


fig = plt.figure(figsize = (20, 6))
full['runtime'].fillna(full['runtime'].mean(), inplace=True)
sns.distplot(full['runtime'])
fig.suptitle('Runtime (full)', fontsize=20)
plt.show()


# ## Year

# In[ ]:


plt.figure(figsize=(16, 8))
plt.bar(train['release_date_year'], train['revenue'], label='revenue')
plt.bar(train['release_date_year'], train['budget'], label='budget')
plt.title('Revenue/Budget per year')
plt.xlabel('Year')
plt.ylabel('Budget')
plt.legend(loc='upper center', frameon=False)
plt.show()


# In[ ]:


plt.figure(figsize=(12, 8))
plt.bar(train['release_date_month'], train['revenue'], label='revenue')
plt.bar(train['release_date_month'], train['budget'], label='budget')
plt.title('Revenue/Budget per Month')
plt.xlabel('Month')
plt.ylabel('Revenue / Budget')
plt.legend(loc='upper center', frameon=False)
plt.show()


# In[ ]:


plt.figure(figsize=(20, 8))
plt.bar(train['release_date_year'], train['popularity'], alpha=0.5)
plt.bar(test['release_date_year'], test['popularity'], alpha=0.5)
plt.title('Popularity per year')
plt.xlabel('Popularity (full)')
plt.ylabel('Year')
plt.show()


# ## Genres

# In[ ]:


plt.figure(figsize=(16, 8))
ax = sns.barplot(x='Counts', y='Value', data=genres_unique, palette='Spectral')
ax.set_title(label='Distribution of genres')
ax.set_ylabel('')
ax.set_xlabel('Number of movies')
plt.show()


# ## Top companies

# In[ ]:


companies_unique = uniqueValues(full, 'production_companies', 'name').sort_values(by='Counts', ascending=False)

TOP_COMPANIES = 15
plt.figure(figsize=(16, 8))
ax = sns.barplot(x='Counts', y='Value', data=companies_unique[:TOP_COMPANIES], palette='Spectral')
ax.set_title(label='Distribution of top {} companies'.format(TOP_COMPANIES))
ax.set_ylabel('')
ax.set_xlabel('Number of movies')
plt.show()


# In[ ]:


prodc_unique = uniqueValues(full, 'production_countries', 'iso_3166_1').sort_values(by='Counts', ascending=False)

TOP_COUNTRIES = 15
plt.figure(figsize=(12, 6))
ax = sns.barplot(y='Counts', x='Value', data=prodc_unique[:TOP_COUNTRIES], palette='hot')
ax.set_title(label='Distribution of top {} production countries'.format(TOP_COUNTRIES))
ax.set_ylabel('')
ax.set_xlabel('')
plt.show()


# ## Top languages

# In[ ]:


TOP_LANGUAGES = 15
plt.figure(figsize=(12, 6))
ax = sns.barplot(y='Counts', x='Value', data=languages_unique[:TOP_LANGUAGES], palette='hot')
ax.set_title(label='Distribution of top {} languages'.format(TOP_LANGUAGES))
ax.set_ylabel('')
ax.set_xlabel('')
plt.show()


# ## Cast

# In[ ]:


cast_unique = uniqueValues(full, 'cast', 'name').sort_values(by='Counts', ascending=False)

TOP_CAST = 25
plt.figure(figsize=(16, 8))
ax = sns.barplot(x='Counts', y='Value', data=cast_unique[:TOP_CAST], palette='BuPu_r')
ax.set_title(label='Distribution of top {} actors'.format(TOP_CAST))
ax.set_ylabel('')
ax.set_xlabel('Number of movies')
plt.show()


# In[ ]:


cast_unique = uniqueValues(full, 'cast', 'gender')

colors = [ '#F2B134', '#068587', '#ED553B']
labels=['Gender 0', 'Gender 1', 'Gender 2']
fig, ax = plt.subplots(figsize=(8, 6))
ax.pie(cast_unique['Counts'], labels=labels, colors=colors, autopct='%1.1f%%')
ax.axis('equal')
ax.set_title(label='Distribution of genders in actors')
plt.show()


# ## Keywords

# In[ ]:


keywords_unique = uniqueValues(full, 'Keywords', 'name').sort_values(by='Counts', ascending=False)

TOP_COMPANIES = 25
plt.figure(figsize=(16, 8))
ax = sns.barplot(x='Counts', y='Value', data=keywords_unique[:TOP_COMPANIES], palette='icefire_r')
ax.set_title(label='Most used Keywords')
ax.set_ylabel('')
ax.set_xlabel('')
plt.show()


# - **duringcreditstinger, aftercreditstinger**: [Post-credits scene](https://en.wikipedia.org/wiki/Post-credits_scene)

# ## Most used words

# In[ ]:


plt.figure(figsize = (12, 12))
text = ' '.join(train['overview'].fillna('').values)
wordcloud = WordCloud(margin=10, background_color='white', colormap='Greens', width=1200, height=1000).generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.title('Top words in overview', fontsize=20)
plt.axis('off')
plt.show()


# In[ ]:


plt.figure(figsize = (12, 12))
text = ' '.join(train['title'].fillna('').values)
wordcloud = WordCloud(margin=10, background_color='white', colormap='Reds', width=1200, height=1000).generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.title('Top words in titles', fontsize=20)
plt.axis('off')
plt.show()


# ## Missing Values

# In[ ]:


def getNullCols(df):
    total_null = df.isnull().sum().sort_values(ascending=False)
    percent_null = ((df.isnull().sum() / df.isnull().count()) * 100).sort_values(ascending=False)
    missing_data = pd.concat([total_null, percent_null], axis=1, keys=['Total', 'Percent'])
    return missing_data
null_df = getNullCols(train_data).head(10)

plt.figure(figsize=(10, 5))
sns.barplot(y=null_df.index, x=null_df['Total'], palette='icefire_r')
plt.title('Total null values by feature')
plt.xlabel('')
plt.ylabel('')
plt.show()
display(null_df)


# ![](https://static.wixstatic.com/media/0b319a_d13584c8485e40d680d12e2f54793feb~mv2.jpg/v1/fill/w_504,h_283,al_c,lg_1,q_80/0b319a_d13584c8485e40d680d12e2f54793feb~mv2.jpg)

# In[ ]:




