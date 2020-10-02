#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # Playing next...

# Spotify apps have a feature called "Made for you" which shows a few playlists curated upon the songs that you already liked and added to your library. In this notebook, we will try to build a simple recommender using K-Means clustering to predict which song should play after the one that you've already selected.

# ### Loading the data

# This is a dataset containing Top 50 songs on 2019, along with a few features. Let's have a look at it:

# In[ ]:


data = pd.read_csv("../input/top50spotify2019/top50.csv",encoding='ISO-8859-1')
data.head()


# Let's rename the labels for better understanding:

# In[ ]:


data.rename(columns={"Beats.Per.Minute":"BPM",
                     "Valence.":"Valence",
                     "Acousticness..":"Acousticness",
                     "Loudness..dB..":"Loudness",
                     "Speechiness.":"Speechiness",
                     "Track.Name":"Track",
                     "Artist.Name":"Artist"},inplace=True)


# I'll drop the `Unnamed: 0` and `Length.` features because it seems irrelevant for a recommender system:**

# In[ ]:


data.drop(["Unnamed: 0","Length."], axis=1,inplace=True)

data.head()


# A lot of these genres are variations of the 'pop' genre. It would be easier to just club them all together into one genre called 'pop':

# In[ ]:


import re 

for genre in data['Genre']:
    if re.search('pop', genre):
        data.loc[data.Genre == genre, 'Genre'] = 'pop'
        
data.head()


# Now let's encode all genres using Label Encoding:

# In[ ]:


from sklearn.preprocessing import LabelEncoder 
  
le = LabelEncoder() 
  
data['Genre']= le.fit_transform(data['Genre']) 

data.head()


# ### Building the recommender model

# Let's remove the track name and artist name (trivial to the track) from the data and store the features of the song in another variable:

# In[ ]:


X = data.copy()
X.drop(["Track","Artist"], axis=1,inplace=True)
X.head()


# Since we are using K-Means clustering, we need to figure out the number of clusters we wish to divide our data into. We use the elbow method to find the optimal number of clusters. 

# In[ ]:


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 29)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# The optimal number of clusters seems to be 5, so let us proceed with that.

# In[ ]:


kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 29)
y_kmeans = kmeans.fit_predict(X)


# Our cluster labels are now stored in y_kmeans. We will add that as a column to our `data` variable:

# In[ ]:


data['ClusterID'] = y_kmeans
data.head()


# Our curated playlist generator is now ready.

# ### Implementation

# Spotify's playlist recommendations look something like this:
# 
# 

# ![My daily playlist recommendation](https://rainnews.com/wp-content/uploads/2016/09/spotify-daily-mix.png)

# We have done the same to curate the Top 50 songs in our dataset into 5 playlists based on the features of the song, i.e. genre, energy, danceability, loudness etc.

# In[ ]:


data.sort_values(by=['ClusterID'], inplace=True)

for clusternumber in data.ClusterID.unique():
    print("\nPlaylist Number: " + str(clusternumber+1))
    print(data.loc[data.ClusterID == clusternumber, ['Track', 'Artist']])


# ### Thank you for reading! If you liked my notebook, please leave an upvote!
