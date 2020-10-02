#!/usr/bin/env python
# coding: utf-8

# <img src="https://charts-images.scdn.co/REGIONAL_GLOBAL_DEFAULT.jpg">
# 
# ## Importing the libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# # Reading the data

# In[ ]:


data=pd.read_csv('/kaggle/input/top50spotify2019/top50.csv',encoding='ISO-8859-1')


# In[ ]:


data.head()


# In[ ]:


print("The shape of the data is ",data.shape)


# In[ ]:


print("The columns are: ")
print(data.columns)


# In[ ]:


print(data['Unnamed: 0'])


# So the column Unnamed: 0 is index so we can drop that column

# In[ ]:


data.drop(['Unnamed: 0'],axis=1,inplace=True)


# 
# # About Columns
# ** Track.NameName of the Track <br>
#  Artist.NameName of the Artist <br>
#  Genrethe genre of the track <br>
#  Beats.Per.MinuteThe tempo of the song.<br>
#  EnergyThe energy of a song - the higher the value, the more energtic. song<br>
#  DanceabilityThe higher the value, the easier it is to dance to this song.<br>
#  Loudness..dB..The higher the value, the louder the song.<br>
#  LivenessThe higher the value, the more likely the song is a live recording.<br>
#  Valence.The higher the value, the more positive mood for the song.<br>
#  Length.The duration of the song.<br>
#  Acousticness..The higher the value the more acoustic the song is.<br>
#  Speechiness.The higher the value the more spoken word the song contains.<br>
#  Popularity :The higher the value the more popular the song is **

# ## Sorting with respect to Popularity (More the value more popular is the song)

# In[ ]:


data.sort_values(axis=0,ascending=False,inplace=True,by='Popularity')


# In[ ]:


data.head()


# In[ ]:


data.describe()


# >> Checking for the null data

# In[ ]:


data.isnull().sum()


# ### there is no null data in any column

# In[ ]:


plt.figure(figsize=(10,10))
correlation=data.corr(method='spearman')
plt.title('Correlation heatmap')
sns.heatmap(correlation,annot=True,vmin=-1,vmax=1,center=1)


# # Trackname and popularity

# In[ ]:


track_name=data['Track.Name'].value_counts()
track_name[:10]


# In[ ]:


track_name[39:49]


# > Here we can see that the track names of top 10 and 40-50 have a very comman trend that is the names are the *Commanly used words like : Shallow , bad guy, happier, Ransom,takeaway, etc *
# >> It is not necessary that the name should be unique even the very commanly used words of okay

# # Artist name and Popularity
# 

# <img src="https://i2.wp.com/metro.co.uk/wp-content/uploads/2019/12/GettyImages-1160891166.jpg?quality=90&strip=all&zoom=1&resize=644%2C429&ssl=1">

# In[ ]:


artist_name=data['Artist.Name'].value_counts()
artist_name[:20]


# In[ ]:


plt.figure(figsize=(18,9))
sns.barplot(x=artist_name[:20],y=artist_name[:20].index)
plt.title("Artist name")


# Yes the artist will definitely effect the popularity of the song  
# > More popular the artist is more the are the chances that his song will become Popular

# # Genre and Popularity

# In[ ]:


genre=data['Genre'].value_counts()
genre


# In[ ]:


plt.figure(figsize=(18,9))
sns.barplot(x=genre[:10],y=genre[:10].index)
plt.title("Genre")


# >> 8 in 50 popular songs are pop
# >>> Is pop and dance pop a same thing????

# # Beats per minute of popular songs

# ### The song with highest bpm is thousand  peaking at approximately 1,015 BPM.

# In[ ]:


plt.figure(figsize=(12,8))
sns.regplot(x='Beats.Per.Minute', y='Popularity',ci=None, data=data)
sns.kdeplot(data['Beats.Per.Minute'],data.Popularity)
plt.title("BPM and Popularity")


# In[ ]:


beats=data['Beats.Per.Minute']
print("min :",beats.min())
print("max :",beats.max())
print("mean :",beats.mean())


# The above graph is very dark for 100 bpm.
# 

# #  Energy and Popularity

# In[ ]:


plt.figure(figsize=(12,8))
sns.regplot(x='Energy', y='Popularity',ci=None, data=data)
sns.kdeplot(data.Energy,data.Popularity)


# In[ ]:


energy=data['Energy']
print("min :",energy.min())
print("max :",energy.max())
print("mean :",energy.mean())


# > ### the dots are more concentrated between 60 to 90(approx)
# 

# # Danceability and popularity

# In[ ]:


plt.figure(figsize=(12,8))
sns.regplot(x='Danceability', y='Popularity',ci=None, data=data)
sns.kdeplot(data.Danceability,data.Popularity)


# > The danceability is more concntrated in 50-100 <br>
# > More danceability < 45 are only 2 songs in 50
# >> This means if the danceability in song is more than 50% than the chances of that song becomming popular are more

# # Loudness and Popularity

# Loudness is given in db
# #### Higher the value more louder is the song
# ### The highest value for loudness of the song is 0.0db in spotify
# 
# 
# ## <a href="https://open.spotify.com/album/3w9Sw9Ocws2diPLMkskcHj"> Listern to the Loudest song on Spotify:

# In[ ]:


plt.figure(figsize=(12,8))
sns.regplot(x='Loudness..dB..', y='Popularity',ci=None, data=data)
sns.kdeplot(data['Loudness..dB..'],data.Popularity)


# ### How spotify calculates the Loudness : https://artists.spotify.com/faq/mastering-and-loudness#how-does-spotify-calculate-loudness

# In[ ]:


loudness=data['Loudness..dB..']
print("min :",loudness.min())
print("max :",loudness.max())
print("mean :",loudness.mean())


# ### Many songs have the loudness of nearly -6db

# # Liveness and Popularity

# #### The higher the value, the more likely the song is a live recording

# In[ ]:


plt.figure(figsize=(12,8))
sns.regplot(x='Liveness', y='Popularity',ci=None, data=data)
sns.kdeplot(data['Liveness'],data.Popularity)


# In[ ]:


liveness=data['Liveness']
print("min :",liveness.min())
print("max :",liveness.max())
print("mean :",liveness.mean())


# ### Most of the songs have liveness less than 20% 
# 
# songs which are not live recording are more popular

# # Valence and Popularity
# #### Valence.The higher the value, the more positive mood for the song

# In[ ]:


plt.figure(figsize=(12,8))
sns.regplot(x='Liveness', y='Popularity',ci=None, data=data)
sns.kdeplot(data['Liveness'],data.Popularity)


# ### Most of the values are between 0 to 20

# # Length and Popularity

# ### The longest song is <i> The Rise and Fall of Bossanova </i> length = 16212 seconds ie. 13 hours 23 mins 32 seconds

# In[ ]:


plt.figure(figsize=(12,6))
sns.distplot(data['Length.'])


# In[ ]:


length=data['Length.']
print('Max :',length.max())
print('Min :',length.min())
print("Mean :",length.mean())


# In[ ]:


arr=[x for x in range(100,321,20)]
arr=tuple(arr)
length_cat=pd.cut(length,arr)


# ### Converting to categorical data

# In[ ]:


length_counts=length_cat.value_counts()
print(length_counts)


# In[ ]:


plt.figure(figsize=(18,9))
sns.barplot(x=length_counts,y=length_counts.index)


# ### Most of the songs are in the range of 2 min 40 sec and 4 min

# # Acousticness and Popularity

# In[ ]:


plt.figure(figsize=(12,8))
sns.jointplot(x='Acousticness..', y='Popularity',kind='kde', data=data)


# # Speechiness. and Popularity

# ### The higher the value the more spoken word the song contains

# In[ ]:


plt.figure(figsize=(12,8))
sns.regplot(x='Speechiness.', y='Popularity',ci=None, data=data)
sns.kdeplot(data['Speechiness.'],data.Popularity)


# In[ ]:


Speechiness=data['Speechiness.']
print("min :",Speechiness.min())
print("max :",Speechiness.max())
print("mean :",Speechiness.mean())


# ## The songs with less spoken words are more popular

# #### Please upvote if you like this notebook
