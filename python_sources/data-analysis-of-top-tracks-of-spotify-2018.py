#!/usr/bin/env python
# coding: utf-8

# At the end of each year, Spotify compiles a playlist of the songs streamed most often over the course of that year. This year's playlist (Top Tracks of 2018) includes 100 songs. The question is: What do these top songs have in common? Why do people like them?

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


df = pd.read_csv("../input/top-spotify-tracks-of-2018/top2018.csv")


# In[ ]:


df.info()


# In[ ]:


df.describe()


# In[ ]:


df.head()


# In[ ]:


## Initialize the figure
plt.style.use('seaborn-darkgrid')
plt.figure(figsize=(20,8))
df["danceability"].plot(color="blue")
plt.xlabel('Top 100 Tracks',fontsize=20)
plt.ylabel('danceability',fontsize=20)
plt.title('danceability Plot',fontsize=20)


# In[ ]:


#Let's plot all features for top 100 tracks
df_features_list = ['energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness',
       'valence', 'tempo', 'duration_ms', 'time_signature']

for i in df_features_list:
    ## Initialize the figure
    plt.style.use('seaborn-darkgrid')
    plt.figure(figsize=(20,8))
    df[i].plot(color="blue")
    plt.xlabel('Top 100 Tracks',fontsize=20)
    plt.ylabel(i,fontsize=20)
    plt.title(i +' Plot',fontsize=20)
    


# In[ ]:


sns.heatmap(df.corr())


# Top artists in Top 100

# In[ ]:


df['artists'].value_counts().head(10)

