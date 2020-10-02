#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ### **THE MOST POPULAR SONGS ON SPOTIFY**
# 
# Content
# There are the most popular songs in the world by year and 13 variables to be explored. Data were stracted from: http://organizeyourmusic.playlistmachinery.com/
# 
# Variables description:
# 
# * title: Song's title
# * artist: Song's artist
# * top genre: the genre of the track
# * year: Song's year in the Billboard
# * bpm: Beats.Per.Minute - The tempo of the song.
# * nrgyEnergy: The energy of a song - the higher the value, the more energtic. song
# * dnce: Danceability - The higher the value, the easier it is to dance to this song.
# * dBLoudness..dB..: The higher the value, the louder the song
# * live: Liveness - The higher the value, the more likely the song is a live recording
# * val: Valence - The higher the value, the more positive mood for the song.
# * dur: Length - The duration of the song.
# * acous: Acousticness.. - The higher the value the more acoustic the song is.
# * spch: Speechiness - The higher the value the more spoken word the song contains.
# * pop: Popularity- The higher the value the more popular the song is.

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
get_ipython().run_line_magic('config', "InlineBackend.figure_format ='retina'")
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv('/kaggle/input/top-spotify-songs-from-20102019-by-year/top10s.csv', encoding='ISO-8859-1', index_col = 0)
df.head()
df.shape
df.dtypes
df.describe()


# In[ ]:


df.columns


# In[ ]:


# checking distributions of numeric values

# histogram of whole ds
fig = plt.figure(figsize = (12,10))
ax = fig.gca()
df.hist(ax = ax)
plt.style.use('ggplot')


# In[ ]:


# plotting the correlation between population and the other numeric features
g = sns.pairplot(df, y_vars="pop", x_vars=['bpm', 'nrgy', 'dnce', 'dB','live', 'val', 'dur', 'acous', 'spch'])
g.fig.set_figheight(3)
g.fig.set_figwidth(15)


# In[ ]:


# Correlation Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True,cmap = sns.cm.rocket_r)


# In[ ]:




