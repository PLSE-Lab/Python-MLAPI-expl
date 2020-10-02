#!/usr/bin/env python
# coding: utf-8

# **Import libraries**

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O 
import matplotlib.pyplot as plt
import missingno as msno


# **Reading csv file**

# In[ ]:


song_data = pd.read_csv("../input/top-spotify-songs-from-20102019-by-year/top10s.csv", encoding='ISO-8859-1', index_col=0)


# In[ ]:


type(song_data)


# **Display first 10 rows from dataset**

# In[ ]:


song_data.head()


# **Number of rows and columns**

# In[ ]:


song_data.shape


# **Display dataset columns**

# In[ ]:


song_data.columns


# **Renaming column names**

# In[ ]:


song_data.rename(columns = {'bpm':'bits_per_min', 'nrgy':'energy','dnce':'danceability', 'dB':'loudness(dB)', 'live':'liveness', 'val':'valence', 'dur':'duration', 'acous':'acousticness', 'spch':'speechiness', 'pop':'popularity'}, inplace=True)
song_data.columns


# In[ ]:


song_data.tail()


# **Overview**

# In[ ]:


fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(16,10))
plt.tight_layout()
axes[0][0].hist(song_data["bits_per_min"],bins=20)
axes[0][0].set_title("bits_per_min")
axes[0][1].hist(song_data["energy"],bins=20)
axes[0][1].set_title("energy")
axes[0][2].hist(song_data["popularity"],bins=20)
axes[0][2].set_title("popularity")
axes[1][0].hist(song_data["danceability"],bins=20)
axes[1][0].set_title("danceability")
axes[1][1].hist(song_data["loudness(dB)"],bins=20)
axes[1][1].set_title("loudness(dB)")
axes[1][2].hist(song_data["liveness"],bins=20)
axes[1][2].set_title("liveness")
axes[2][0].hist(song_data["valence"],bins=20)
axes[2][0].set_title("valence")
axes[2][1].hist(song_data["duration"],bins=20)
axes[2][1].set_title("duration")
axes[2][2].hist(song_data["acousticness"],bins=20)
axes[2][2].set_title("acousticness")
axes[3][0].hist(song_data["speechiness"],bins=20)
axes[3][0].set_title("speechiness")


# In[ ]:


fig = plt.figure(figsize=(20,14))
ax_data = fig.gca()
song_data.hist(ax=ax_data)
plt.style.use('ggplot')


# In[ ]:


fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(16,10))
plt.tight_layout()
axes[0][0].plot(song_data["bits_per_min"])
axes[0][0].set_title("bits_per_min")
axes[0][1].plot(song_data["energy"])
axes[0][1].set_title("energy")
axes[0][2].plot(song_data["popularity"])
axes[0][2].set_title("popularity")
axes[1][0].plot(song_data["danceability"])
axes[1][0].set_title("danceability")
axes[1][1].plot(song_data["loudness(dB)"])
axes[1][1].set_title("loudness(dB)")
axes[1][2].plot(song_data["liveness"])
axes[1][2].set_title("liveness")
axes[2][0].plot(song_data["valence"])
axes[2][0].set_title("valence")
axes[2][1].plot(song_data["duration"])
axes[2][1].set_title("duration")
axes[2][2].plot(song_data["acousticness"])
axes[2][2].set_title("acousticness")
axes[3][0].plot(song_data["speechiness"])
axes[3][0].set_title("speechiness")


# **Checking NaN (null) values in the dataset**

# In[ ]:


msno.matrix(song_data)


# **Plotting values to display their range**

# In[ ]:


song_data[['duration','bits_per_min','loudness(dB)','popularity','energy','danceability']].plot(figsize=(20,10))


# In[ ]:


song_data[['speechiness','loudness(dB)','popularity','acousticness']].plot.area(figsize=(20,10))


# **Show dataset information**

# In[ ]:


song_data.info()


# In[ ]:


song_data.describe()


# **Number of artist**

# In[ ]:


song_data['artist'].nunique()


# **Top 10 artist based on their released song**

# In[ ]:


song_data['artist'].value_counts().reset_index().head(10)


# In[ ]:


song_data['artist'].value_counts().head(10).plot.bar(figsize=(20,10))
plt.xlabel('Artist Name')
plt.ylabel('Number of song')
plt.title('Top 10 artist')


# In[ ]:


song_data['artist'].value_counts().head(10).plot.line(figsize=(20,10))
plt.xlabel('Artist Name')
plt.ylabel('Number of song')
plt.title('Top 10 artist')


# In[ ]:


song_data['artist'].value_counts().head(10).plot.pie(figsize=(10,10), autopct='%1.0f%%')
plt.title('Top 10 artist based on song in percentage')


# **Count of top genre of songs**

# In[ ]:


song_data['top genre'].unique()


# In[ ]:


song_data['top genre'].nunique()


# **Number of songs in each genre**

# In[ ]:


song_data['top genre'].value_counts().head(10)


# In[ ]:


song_data['top genre'].value_counts().plot(figsize=(20,10))
plt.xlabel('Top genre Name')
plt.ylabel('Number of song')
plt.title('Top 10 genre')


# In[ ]:


song_data['top genre'].value_counts().head(10).plot.bar(figsize=(20,10))
plt.xlabel('Top genre Name')
plt.ylabel('Number of song')
plt.title('Top 10 genre bar plot')


# In[ ]:


song_data['top genre'].value_counts().head(10).plot.pie(figsize=(10,10),autopct='%1.1f%%')
plt.title('Top 10 genre in percentage')


# **Song list of Ed Sheeran**

# In[ ]:


ed_she = song_data[song_data['artist']=='Ed Sheeran']
ed_she


# **Drop columns from Dataset**

# In[ ]:


ed_she_new = ed_she.drop(['year'],axis=1)
ed_she_new


# **Overview of Ed Sheeran Data**

# In[ ]:


fig = plt.figure(figsize=(20,14))
ed_data = fig.gca()
ed_she_new.hist(ax=ed_data, bins=50)
plt.style.use('ggplot')


# In[ ]:


ed_she_new.plot.line(figsize=(20,10))
plt.title('Plotting Ed Shreen data')


# In[ ]:


ed_she_new.plot.area(figsize=(20,10))
plt.title('Area Plotting of Ed Shreen data')


# In[ ]:


ed_she_new.plot.bar(figsize=(20,10))
plt.xlabel('Song id')
plt.ylabel('Song values')
plt.title('Bar Plotting of Ed Shreen data')


# In[ ]:


ed_she_new.plot.hist(bins=50,figsize=(20,10))


# **Number of song in each year**

# In[ ]:


song_data['year'].value_counts()


# In[ ]:


song_data['year'].value_counts().plot.bar(figsize=(20,10))
plt.xlabel('Year')
plt.ylabel('Number of Song')
plt.title('Number of songs in each year')


# In[ ]:


song_data['year'].value_counts().plot.pie(figsize=(10,10),autopct='%1.1f%%')


# **Number of songs of Katy Perry in Each year**

# In[ ]:


song_data[song_data['artist']=='Katy Perry']['year'].value_counts()


# In[ ]:


song_data[song_data['artist']=='Katy Perry']['year'].value_counts().plot.bar(figsize=(20,10))
plt.xlabel('Year')
plt.ylabel('Number of Song')
plt.title('Bar Plotting of Katy Perry song in each year')


# **Which artist do the pop song**

# In[ ]:


song_data[song_data['top genre']=='pop']['artist'].unique()


# In[ ]:


song_data[song_data['top genre']=='pop']['artist'].nunique()


# In[ ]:


song_data[song_data['top genre']=='pop']['artist'].value_counts()


# In[ ]:


song_data[song_data['top genre']=='pop']['artist'].value_counts().plot.bar(figsize=(20,10))
plt.xlabel('Artist name')
plt.ylabel('Number of Pop Song')
plt.title('Bar Plotting of Pop song based on artist')


# In[ ]:


song_data[song_data['top genre']=='pop']['artist'].value_counts().plot.pie(figsize=(10,10),autopct='%1.1f%%')
plt.title('Plotting of Pop song based on artist in percentage')


# **Top 10 songs of all time based on liveness**

# In[ ]:


song_data.sort_values(by='liveness',ascending=False).head(10)[['liveness','title','year']]


# **Top 10 songs of all time based on popularity**

# In[ ]:


song_data.sort_values(by='popularity',ascending=False).head(10)[['popularity','title','year']]

