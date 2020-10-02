#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
from scipy.stats import pearsonr
get_ipython().run_line_magic('matplotlib', 'inline')
import os
print(os.listdir("../input"))


# In[ ]:


top2018= pd.read_csv('../input/top2018.csv')
top2018.shape


# In[ ]:


top2018.head()


# In[ ]:


def convertMillis(millis):
     minutes=(millis/(1000*60))%60
     return minutes
top2018['duration_min']=convertMillis(top2018['duration_ms'])
top2018.drop('duration_ms', axis=1, inplace=True)


# Lets idendify the corelation between the columns
# 

# In[ ]:


sns.heatmap(top2018.corr(), cmap= 'CMRmap')


# **We can see that there is a definite relationship between loudness and energy
# **

# Let's see who are the 20 most played artists in the year 2018

# In[ ]:


top2018['artists'].value_counts().head(20)


# 
# We can see the XXXTENTACION and Post Malone are the most famous artists with 6 songs in the Top 100 list.

# **Danceability****

# In[ ]:


sns.set_style(style='dark')
sns.distplot(top2018['danceability'],hist=True,kde=True)


# In[ ]:


sns.boxplot(data=top2018['danceability'])


# We can see that the distribution of daceability is skewed towards 1. With the average being somewhere around 0.7. So we can see that most of the songs in the list are danceable. Which can mean that the people listen to more songs that are danceable rather than any other kind. 

# **Positivity in the Top 25 songs**
# Valence is a measure of how positive the track is and a score more than 0.6-0.7 represents a happy track, Let us see how positive the top tracks are . 
# 

# In[ ]:


top25= top2018.iloc[0:25, :]


# In[ ]:


sns.distplot(top2018['valence'])


# * We can see that most of the songs are of valence 0.4 which can be termed as neutral songs. But the overall distribution is curved towards being positive. 

# In[ ]:


happy_songs= pd.DataFrame(top25[top25['valence']>0.5])
happy_songs


# In[ ]:


sns.heatmap(happy_songs.corr(), annot=True)


# In[ ]:


top2018[['name','artists','danceability','energy','valence','tempo']].sort_values(by='valence',ascending=False).head(10)


# So I kinda expected Ed sheeran to be there , because his songs are mostly on a positive note, bruno masrs with finesse is also there. Interesting to see that all these songs have a very high  danceability. But they are not very fast, the tempo seems to be in the range of 80-100 with the only exception of Piso 21 song (it has a comparatively low danceability). 
# 

# From this we can see that in the top songs there is relationship between the Tempo of a song and the positivity of hte songs, a high tempo means that the song is going to be a happy songs. 
# A tempo from 110 to 160 can be termed as an Allegro.

# > * **Live Songs**:
# Let us see how many of these songs were performed in front of a live audience. The Liveness variable with a value higher than 0.8 signifies that the song was performed live.

# In[ ]:


live_songs = top2018[top2018['liveness']>0.6]
live_songs


# But it seems that there are almost no live songs in the top song list. Only one songs has a liveness factor greter than 0.6. 

# In[ ]:




