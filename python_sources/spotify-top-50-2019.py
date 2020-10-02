#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import re 

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # I love Spotify.
# The Top 50 playlist is one I visit often when I need to find new music. About half the time, I end up liking a song from the playlist a lot, otherwise I simply detest it. Now, my music taste varies between alt-pop and I generally prefer songs that are almost happy but sad at the same time. But does the whole world share the same views as me, or am I just an outlier? 
# Let's check out this dataset to find out further.

# In[ ]:


data = pd.read_csv("../input/top50spotify2019/top50.csv",encoding='ISO-8859-1')
data.head()


# ### Features
# 
# * **Danceability** = Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity.
# * **Energy** = Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a Bach prelude scores low on the scale. Perceptual features contributing to this attribute include dynamic range, perceived loudness, timbre, onset rate, and general entropy.
# * **Liveness** = Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live.
# * **Loudness** = The overall loudness of a track in decibels (dB). Loudness values are averaged across the entire track and are useful for comparing relative loudness of tracks. Loudness is the quality of a sound that is the primary psychological correlate of physical strength (amplitude).
# * **Speechiness** = Detects the presence of spoken words in a track.

# Let's rename the columns for better readablity and delete the unnecessary "Unnamed: 0" column as well as the ones that we will not be analysing:

# In[ ]:


data.rename(columns={"Loudness..dB..":"Loudness",
                     "Speechiness.":"Speechiness",
                     "Track.Name":"Track",
                     "Artist.Name":"Artist"},inplace=True)
data.drop(["Unnamed: 0",
          "Valence.", 
          "Acousticness..",
          "Beats.Per.Minute", 
          "Length."], axis=1,inplace=True)

data.head()


# Low energy songs like Senorita may also have a higher danceablity factor, while catchy songs like Ariana's 'boyfriend' may have high energy and low danceablity. We shall combine these two attributes into a feature called **"Usability"**, i.e. is someone likely to sing or dance to  this song.

# In[ ]:


data['Usability'] = np.sqrt((data['Energy'])**2 + (data['Danceability'])**2)
data.drop(["Energy", "Danceability"], axis=1,inplace=True)
data.head()


# A lot of these genres are variations of the 'pop' genre. It would be easier to just club them all together into one genre called 'pop'

# In[ ]:


datacopy = data

for genre in data['Genre']:
    if re.search('pop', genre):
        data.loc[data.Genre == genre, 'Genre'] = 'pop'
        
data.head()


# In[ ]:


plt.figure(figsize=(15,10))
sns.scatterplot(y = "Popularity", x = 'Usability',
                hue = "Genre", data = data);


# From the above graph, we can make out that while pop songs are popular regardless of how usable they are, the other genres often have to step up on their usability points to become more popular. A low energy EDM song which you can't dance to is unlikely to be popular enough for Spotify's Top 50. Perhaps you can find it in "Relaxing Studying Music" playlists instead?

# Let's look at the graph for Speechiness vs Popularity for these pop songs

# In[ ]:


plt.figure(figsize=(15,10))
mymodel = np.poly1d(np.polyfit(x = data['Speechiness'], y = data["Popularity"], deg = 4))
myline = np.linspace(1, 50, 100)
plt.plot(myline, mymodel(myline))

sns.regplot(y = "Popularity", x = 'Speechiness', data = data.loc[data['Genre'] == 'pop'], fit_reg = True);


# From the above graph it seems like low speechiness on pop songs does not affect their popularity (maybe when the song is more music and less lyrics). However, when the speechiness increases, the song popularity goes up (maybe because of easy to sing lyrics), and then falls down (maybe because excess lyrics mean less catchiness of the song).

# ## Conclusion
# 

# It seems like catchy pop songs that are easy to sing or fun to dance to will continue to dominate the Top 50 charts on Spotify. For other genres such as EDM, high energy or high danceability is a must if they wish to gain a spot on the top charts, while facing fierce competition form reggaetton songs, which are usually very high on danceability levels.

# ### If you liked this notebook, please leave an upvote and/or a comment! Thanks for reading!

# In[ ]:




