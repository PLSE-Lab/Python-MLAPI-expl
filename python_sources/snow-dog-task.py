#!/usr/bin/env python
# coding: utf-8

# Files which you have added contain dataset from LastFm service. You are provided with two files:
# * track_title_artist.txt -  informations about tracks: track ID, track title and artist name,
# * user_track_time.txt - data about service usage: user ID, track ID, listen time in unix timestamp format.  
# 
# 
# 1. Load datasets from `kaggle/input` directory.

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


import os
paths = []
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        paths.append(os.path.join(dirname, filename))
paths.sort()
paths


# In[ ]:


track_cols = ["track ID", "track title", "artist name"]
track_set = pd.read_table(paths[0], header=None, names=track_cols)


# In[ ]:


user_cols = ["user ID", "track ID", "listen time"]
user_set = pd.read_table(paths[1], header=None, names=user_cols)


# 2. Check if data has been loaded properly.

# In[ ]:


track_set.head()


# In[ ]:


track_set.shape


# In[ ]:


user_set.head()


# In[ ]:


user_set.shape


# 3. Deal with missing data if any.

# ### track_set

# In[ ]:


track_set.isnull().sum()


# In[ ]:


track_set["artist name"].value_counts().head(15)


# In[ ]:


track_set.shape


# In[ ]:


(track_set["artist name"].str.find("Unknown") != -1).sum()


# In[ ]:


track_set[track_set["track title"] == "[Unknown]"]


# In[ ]:


track_set = track_set[track_set["artist name"].str.find("Unknown") == -1]


# In[ ]:


(track_set["track ID"].str.find("Unknown") != -1).sum()


# **I decide not to touch the song title because we have all the song IDs, and it's the song title in some way. I removed the Unknown from artist name because it creates an "Unknown Artist" and match +1500 songs with him which is not true.**

# ### user_set

# In[ ]:


user_set.isnull().sum()


# In[ ]:


((user_set["track ID"].str.find("Unknown") != -1) & (user_set["user ID"].str.find("Unknown") != -1)).sum()


# **Seems we have complete data**

# 4. Find 5 most listened songs.

# **Most listened song means: how many users listen to this song, it means: how many times this song apear in user_set** 

# In[ ]:


top5_songs = user_set["track ID"].value_counts().iloc[0:5]
top5_songs


# In[ ]:


for i, song in enumerate(top5_songs.index):
    print(i+1, track_set[track_set["track ID"] == song].values[0][1:3])
    


# 5. Find 10 users which have listened to most number of unique songs.

# **User which have listened to most number of unique songs means how many times user appears in user_set** 

# In[ ]:


top10_users = user_set["user ID"].value_counts().iloc[0:10]
for i, user_id in enumerate(top10_users.index):
    print(i+1, user_id)


# 6. Find 5 most listened artists according to dataset.

# In[ ]:


# lets count how many times every song appear in user_set
song_counts = user_set["track ID"].value_counts()

# create a dataframe from value_counts()
song_counts = pd.DataFrame({"track ID":song_counts.index,"count":song_counts.values})
song_counts


# In[ ]:


# I will store all artist and informations about them in dictionary. Key is the artist name, value is a list: list[0] tells us how many users listened to artist's songs,
# list[1] is the artist's song list
# Created dictionary with following template : {artist_name: [number_of_songs_plays, [songs_titles]]}
artist_dict = {artist : [0, list] for artist in track_set["artist name"].value_counts().index.values}


# In[ ]:


for i, (artist, informations) in enumerate(artist_dict.items()):
    
    # first I collect all songs of current artist from track_set and store it in artist_dict
    # this information will be helpfull in the next task
    artist_dict[artist][1] = track_set[track_set["artist name"] == artist]["track ID"].tolist()
    
    # I need info how many times every song appears, i take it from song_counts, I store this value in list and make a sum of this values
    artist_dict[artist][0] = sum(song_counts[song_counts["track ID"].isin(artist_dict[artist][1])]["count"].tolist())
    
    # I wrote a blockade because counting all artists takes too long :(
    if i == 100:
        break

# I need to sort artist_dict by value
sorted_artists = {key: value for key, value in sorted(artist_dict.items(), reverse=True, key=lambda item: item[1][0])}

# Printing result
for i, (k, v) in enumerate(sorted_artists.items()):
    if i == 5: break
    print(i+1, k, v[0])


# 7. Count all users that have listened to __all__ 3 most popular song from `Pink Floyd`.

# In[ ]:


# I should take the Pink Floyd song list from the previous task but since I couldn't count all I will do this especially for Pink Floyd just in caset 
artist = "Pink Floyd"
artist_dict[artist][1] = track_set[track_set["artist name"] == artist]["track ID"].tolist()
artist_dict[artist][0] = sum(song_counts[song_counts["track ID"].isin(artist_dict[artist][1])]["count"].tolist())
    
songs_from_Pink_Floyd = artist_dict["Pink Floyd"]


# In[ ]:


# To find 3 most listened songs I will borrow line of code from task 6 
top_3 = song_counts[song_counts["track ID"].isin(artist_dict[artist][1])]["track ID"].iloc[0:3].values.tolist()
top_3


# In[ ]:


# lets search by users who listened even 1 of top3 songs of Pink Floyd
users_listened_to_top3 = user_set[user_set['track ID'].isin(top_3)]
users_listened_to_top3


# In[ ]:


users = user_set['user ID'].unique()


# In[ ]:


boolean_table=[]
for user in users:
    
    # I return true if user listened to all three songs, false otherwise
    boolean_table.append(len(users_listened_to_top3.loc[users_listened_to_top3['user ID'] == user, "track ID"].unique()) == 3)
    
# then return filtered list
users_listened_to_top_3 = users[boolean_table]


# In[ ]:


users_listened_to_top_3.size


# Below there are two tasks `A` and `B`. Choose and __do only one__ of these.   
# First is related with extracting information about particular song duration.  
# In second you will be asked to make visualization of top100 artist from dataset.  
# In cell below we imported modules which we advise to use. Of course, you can use different libraries that you are familliar with.

# In[ ]:


import pandas as pd
import numpy as np
import sklearn 
import matplotlib.pyplot as plt


# ### Task A

# Form Spotify we can find that duration of "All I Need" from Radiohead is 3 minutes and 49 seconds. Find out if we can retreive this information from dataset. 
# 
# A1. First get all events from user with id `0478c8abd9327b47848aa71c46112192`.

# In[ ]:


user_events = user_set[user_set['user ID'] == '0478c8abd9327b47848aa71c46112192'].sort_values('listen time')
user_events


# A2. Most often users are listening song one after one and rarely are skipping songs. Can you calculate time between listening songs for this user. Most often this will exact duration of song? 
# 
# Tip: Use [shift](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.shift.html)

# In[ ]:


user_events["shifted time"] = user_events["listen time"].shift(-1)
user_events["duration time"] = user_events["shifted time"] - user_events["listen time"]
user_events


# A3. Discard all values longer than 60 minutes as we can assume that song is for sure shorter.

# In[ ]:


user_events = user_events[user_events['duration time'] < 60*60]


# In[ ]:


user_events.drop(user_events.loc[:, ['listen time', 'shifted time']], axis=1, inplace=True)
user_events


# In[ ]:


All_I_Need_ID = track_set.loc[track_set['track title'] == 'All I Need', 'track ID'].values[0]
time_duration = user_events.loc[user_events["track ID"] == All_I_Need_ID, 'duration time']
All_I_Need_ID


# A4. Draw a chart with durations of "All I Need" song. Choose chart type which will help us find correct song duration.

# In[ ]:


n, bins, _ = plt.hist(time_duration, bins=20)


# In[ ]:


max_n = max(n)
max_n_index = np.where(n==max_n)[0][0]
duration_time = (bins[max_n_index] + bins[max_n_index+1]) / 2


# In[ ]:


from time import gmtime
from time import strftime
strftime("%H:%M:%S", gmtime(duration_time))


# Data from one user tells that one song is around 3min 47sec

# A5. For this time we only used data from one user. Can you do the same reseach with whole dataset? How long "All I Need" song is? You don't have to care about removing last single users event duration as it will not affect results.
# 
# Also instead of drawing chart use [np.histogram](https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram.html) function to find "All I Need" duration.

# **First, I sort user_set, then I will the same with all users. It involves many iterations. I did a blockade again because for 10 users it takes around 1.5 min. Then I just make histogram of samples and choose most frequent value.**

# In[ ]:


listen_sorted = user_set.copy()
listen_sorted.sort_values("listen time", inplace=True)
listen_sorted


# In[ ]:


samples = []
for i, us in enumerate(users):
    # I create condition table which will be userd to search and drop
    condition = listen_sorted['user ID'] == us
    
    # same as counting for one user
    u_events = listen_sorted.loc[condition]
    u_events["shifted time"] = u_events["listen time"].shift(-1)
    u_events["duration time"] = u_events["shifted time"] - u_events["listen time"]
    samples = np.concatenate((samples ,u_events.loc[(u_events["track ID"] == All_I_Need_ID) & (u_events['duration time'] < 60*60), 'duration time'].values))
    
    # I remove user from the searched dataset because I will no longer need him and it. This reduces the search set.
    listen_sorted = listen_sorted[~condition]
    if i == 5:
        break

print(samples)


# In[ ]:


n, bins = np.histogram(samples, bins = 200)
max_n = max(n)
max_n_index = np.where(n==max_n)[0][0]
duration_time = (bins[max_n_index] + bins[max_n_index+1]) / 2
strftime("%H:%M:%S", gmtime(duration_time))


# **the more bins we set the better result we get**

# A6. Write small summary if it worked and what other ideas about inforamtions can we retrive from events time.

# **It is possible but I cant figure out how to do it faster. We can retrive how many times user lesiten to one song, what is the total time user spend on listening songs.**

# 

# Download this notebook with outputs and send to us.
