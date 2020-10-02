#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


tv_shows = pd.read_csv('/kaggle/input/tv-shows-on-netflix-prime-video-hulu-and-disney/tv_shows.csv')


# # Data Cleaning
# 1. Drop duplicates based on title

# In[ ]:


tv_shows.drop_duplicates(subset='Title',keep='first',inplace=True)


# 2. Fill nulls with zeros and convert both of them into integers convert both ratings on common scale ie. on 100
#    Since ratings play a huge role we have to process Rotten Tomato ratings and Imdb ratings.

# In[ ]:


tv_shows['Rotten Tomatoes'] = tv_shows['Rotten Tomatoes'].fillna('0%')
tv_shows['Rotten Tomatoes'] = tv_shows['Rotten Tomatoes'].apply(lambda x : x.rstrip('%'))
tv_shows['Rotten Tomatoes'] = pd.to_numeric(tv_shows['Rotten Tomatoes'])

tv_shows['IMDb'] = tv_shows['IMDb'].fillna(0)
tv_shows['IMDb'] = tv_shows['IMDb']*10
tv_shows['IMDb'] = tv_shows['IMDb'].astype('int')


# # Wide to Long Format Conversion
# 
# Plotting of the charts would be easier if we get the 1s and 0s in the columns Netflix,Hulu,Disney+ & Prime Video under a categorical section videos, also there might be cases wherin the same show is present in more than one service.

# In[ ]:


tv_shows_long=pd.melt(tv_shows[['Title','Netflix','Hulu','Disney+','Prime Video']],id_vars=['Title'],var_name='StreamingOn', value_name='Present')
tv_shows_long = tv_shows_long[tv_shows_long['Present'] == 1]
tv_shows_long.drop(columns=['Present'],inplace=True)


# # Merging Datasets
# 
# Merging Long Format dataset with dataset that we started with also need to drop unwanted columns.

# In[ ]:


tv_shows_combined = tv_shows_long.merge(tv_shows, on='Title', how='inner')


# In[ ]:


tv_shows_combined.drop(columns = ['Unnamed: 0','Netflix','Hulu', 'Prime Video', 'Disney+', 'type'], inplace=True)


# # Subsetting
# 
# The datasets with IMDB ratings/ Rotten Tomatoes ratings above 0 needs to be considered for plotting.

# In[ ]:


tv_shows_both_ratings = tv_shows_combined[(tv_shows_combined.IMDb > 0) & tv_shows_combined['Rotten Tomatoes'] > 0]


# # Plotting
# 
# 

# 1. Firstly to get the Service with the most content.

# In[ ]:


tv_shows_combined.groupby('StreamingOn').Title.count().plot(kind='bar')


# 2. Violin charts to gauge the content rating(IMDB) and freshness (Rotten Tomatoes) accross all the streaming service.

# In[ ]:


figure = []
figure.append(px.violin(tv_shows_both_ratings, x = 'StreamingOn', y = 'IMDb', color='StreamingOn'))
figure.append(px.violin(tv_shows_both_ratings, x = 'StreamingOn', y = 'Rotten Tomatoes', color='StreamingOn'))
fig = make_subplots(rows=2, cols=4, shared_yaxes=True)

for i in range(2):
    for j in range(4):
        fig.add_trace(figure[i]['data'][j], row=i+1, col=j+1)

fig.update_layout(autosize=False, width=800, height=800)        
fig.show()


# 3. Scatter plot between IMDB and Rotten tomatoes ratings to get the streaming service that has best of both worlds.

# In[ ]:


px.scatter(tv_shows_both_ratings, x='IMDb',y='Rotten Tomatoes',color='StreamingOn')


# # Inference
# 
# 1. Violin Chart
#     * Hulu, Netflix and Amazon Videos all three have got substantial data in lower end of the ratings.As the content increases so the quality decreases for all three.
#     * Prime Videos have got a denser in the top half on looking at the IMDB and performs ok in freshness.
#     * Disney+ being new has done very well in this area as well.
# 2. Scatter Plot
#     * With this another view, it is quite evident, Amazon Prime performs very well in the fourth quadarant. Which verifies our first inference
# 3. Bar Plot
#     * Amazon Prime wins this race in this one.
# 
# So looking at all three we can conclude Amazon Prime is both about quality and quantity.
#    
#    
