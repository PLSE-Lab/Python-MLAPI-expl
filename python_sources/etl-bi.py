#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from pathlib import Path
import IPython
import IPython.display
from sklearn import datasets, linear_model
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


DATA = Path('../input')
song_atrib = DATA/'spotifyclassification/data.csv'
song_world = DATA/'spotifys-worldwide-daily-song-ranking/data.csv'
song_tracks = DATA/'ultimate-spotify-tracks-db/SpotifyFeatures.csv'


# In[ ]:


df_atrib = pd.read_csv(song_atrib)
df_world = pd.read_csv(song_world)
df_tracks = pd.read_csv(song_tracks)


# In[ ]:


df=pd.DataFrame()
df = pd.merge(df_world, df_tracks, how='left', left_on=['Track Name','Artist'], 
              right_on = ['track_name','artist_name'])


# In[ ]:


df = df.dropna()
df = df.drop(['track_name','artist_name', 'URL'], axis=1)
df = df[df['Date']<'2018-01-01']
df = df.drop_duplicates().reset_index(drop=True)


# In[ ]:


condition= df['Position']<=20
df.where(cond=condition,inplace=True)
condition= df.Region!='global'
df.where(cond=condition,inplace=True)


# In[ ]:


df = df.dropna().reset_index(drop=True)


# In[ ]:


df = df.drop_duplicates(['track_id', 'Region', 'Date']).reset_index(drop=True)


# In[ ]:


df['Region']=df['Region'].str.upper()


# In[ ]:


df['Date'] = pd.to_datetime(df['Date'])
df['Date2'] = df['Date'].dt.strftime('%d%m%Y')


# In[ ]:


df.to_csv('world_df.csv', index=False)


# In[ ]:


df[['Region', 'Streams']].groupby('Region').mean()


# In[ ]:


df[df['Region']=='US']['Track Name'].nunique()


# In[ ]:


df[df['Region']=='AR']['Track Name'].nunique()


# In[ ]:


df[df['Track Name']=='La Pegajosa']


# In[ ]:


df.groupby('Artist')['Streams'].sum()*0.006


# In[ ]:





# In[ ]:




