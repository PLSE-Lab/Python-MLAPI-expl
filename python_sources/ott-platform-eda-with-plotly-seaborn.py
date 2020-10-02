#!/usr/bin/env python
# coding: utf-8

# ![](https://www.wannado.com/wp-content/uploads/2020/02/Whats-New-to-Streaming-Wannado.jpg)

# From adding star power by roping in veteran A-list stars, hosting digital concerts with big names attached, bringing movies onto the streaming services within a few months of theatrical release to reviving cancelled shows on fan demand -- the world of over-the-top or OTT platforms is expanding, and what started on the small screens of the smartphones the world over, suddenly threatens to becoming bigger than the big screen.
# 
# What is significant is that the OTT players are out setting up an all-encompassing entertainment space of their own, one which seems to aim at trying to render all other avenues of entertainment redundant.
# 
# Netflix and Amazon Prime have been producing original shows and films for a while now, which often generate more buzz than much of what is made by traditional sources as movie studios or TV production houses.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from plotly.offline import init_notebook_mode, iplot 
import plotly.graph_objects as go
import plotly.offline as py
import plotly.express as px
import pycountry
py.init_notebook_mode(connected=True)
import folium 
from folium import plugins
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
plt.rcParams['figure.figsize'] = 8, 5
pd.options.mode.chained_assignment = None 
pd.set_option('display.max_columns',None)
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


df = pd.read_csv('/kaggle/input/movies-on-netflix-prime-video-hulu-and-disney/MoviesOnStreamingPlatforms_updated.csv')
df.head()


# In[ ]:


print('Number of rows and columns :',df.shape) # Number of rows and columns


# In[ ]:


df.describe()


# **Observations:**
# * Oldest movie is from 1902.
# * There are a total of 16744 movies in this dataset.
# * Average runtime of all the movies is about 93 minutes.
# * There is an entry with runtime as 1256 minutes.This looks odd. We need to verify this data.

# **Percentage of missing values**

# In[ ]:


percentage_missing_values = round(df.isnull().sum()*100/len(df),2).reset_index()
percentage_missing_values.columns = ['column_name','percentage_missing_values']
percentage_missing_values = percentage_missing_values.sort_values('percentage_missing_values',ascending = False)
percentage_missing_values


# **Observations:**
# * More than 50% of the data is missing for **Rotten Tomatoes** ratings and **Age** restriction columns. 

# ## Bivariate Analysis
# 

# ## Runtime

# In[ ]:


sns.distplot(df['Runtime']);


# Most of the movies have about 90-100 minutes of runtime. Runtime is positively skewed since there are some movies with very large runtime.

# ## IMDB Ratings

# In[ ]:


sns.distplot(df['IMDb']);


# IMDb ratings of the movies roughly follows normal distribution with mean of about 6.5

# ## Movie count by Language

# In[ ]:


movie_count_by_language = df.groupby('Language')['Title'].count().reset_index().sort_values('Title',ascending = False).head(10).rename(columns = {'Title':'Movie Count'})
fig = px.bar(movie_count_by_language, x='Language', y='Movie Count', color='Movie Count', height=600)
fig.show()


# Almost two-third of the movies in this dataset are of English language.

# ## Yearly Movie Count

# In[ ]:


yearly_movie_count = df.groupby('Year')['Title'].count().reset_index().rename(columns = {'Title':'Movie Count'})
fig = px.bar(yearly_movie_count, x='Year', y='Movie Count', color='Movie Count', height=600)
fig.show()


# ## Movies by Country

# In[ ]:


movies_by_country = df.groupby('Country')['Title'].count().reset_index().sort_values('Title',ascending = False).head(10).rename(columns = {'Title':'Movie Count'})
fig = px.bar(movies_by_country, x='Country', y='Movie Count', color='Movie Count', height=600)
fig.show()


# Majority of the movies listed in this dataset are of US origin.

# ## Lengthiest Movies

# In[ ]:


lengthiest_movies = df.sort_values('Runtime',ascending = False).head(10)
fig = px.bar(lengthiest_movies, x='Title', y='Runtime', color='Runtime', height=600)
fig.show()


# ## Digital Platforms

# In[ ]:


digital_platforms = df[['Netflix','Hulu','Prime Video','Disney+']].sum().reset_index()
digital_platforms.columns = ['Platform', 'Movie Count']
digital_platforms = digital_platforms.sort_values('Movie Count',ascending = False)
labels = digital_platforms.Platform
values = digital_platforms['Movie Count']
pie = go.Pie(labels=labels, values=values, marker=dict(line=dict(color='#000000', width=1)))
layout = go.Layout(title='Digital Platforms Movie Share')
fig = go.Figure(data=[pie], layout=layout)
py.iplot(fig)


# Majority of the movies listed in this datset can be found on Amazon Prime.

# ## Top 10 Shows with highest IMDb ratings

# In[ ]:


top_rated_movies = df.sort_values('IMDb',ascending = False).head(10)
fig = px.bar(top_rated_movies, x='Title', y='IMDb', color='IMDb', height=600)
fig.show()


# ## Top 10 Directors with highest Movies

# In[ ]:


top_directors = df.groupby('Directors')['Title'].count().reset_index().rename(columns = {'Title':'Movie Count'}).sort_values('Movie Count',ascending = False).head(10)
fig = px.bar(top_directors, x='Directors', y='Movie Count', color='Movie Count', height=600)
fig.show()


# ## Top Genres

# In[ ]:


top_genres = df.groupby('Genres')['Title'].count().reset_index().rename(columns = {'Title':'Movie Count'}).sort_values('Movie Count',ascending = False).head(10)
fig = px.bar(top_genres, x='Genres', y='Movie Count', color='Movie Count', height=600)
fig.show()


# ## Please do upvote if you like my work. Happy Learning!
