#!/usr/bin/env python
# coding: utf-8

# # Analysis on the Top 50 Most Listened Songs in Spotify
# 
# ## Summary and Conclusion
# ### Composition of the List
#  - Artist with most songs in the list is __Ed Sheeran__ with 4 songs, the second place with 2 songs, and the third place with 1 song. There are only 3 places.
#  - Top 3 genres are Dance pop (8 songs), pop (7 songs), Latin (5 songs)
#  - The most preferred BPMs are from __96 to 137.5 BPM__ (hip hop BPMs).
#  - The most listened songs are mostly _medium-high_ energy songs between __55.25 to 74.75%__ .
#  - The most listened songs are with _medium-high_ danceability of __67 to 79.75 %__ and listeners highly prefer danceability close around __73.5%__.
#  - The most preferred loudness is coincidentially around the _medium-high_ range between __-6.75 dB to -4.00 dB__.
#  - Listeners prefer __studio versions__ more than live version of songs.
#  - Mostly preferred song lengths are from __176.75 to 217.5 minutes__.
#  - Listeners prefer __neutral__ happiness of songs.
#  - Listeners __least prefer acoustic__ songs.
#  - Listeners prefer songs with __fewer words__.
#  - The song must be popular enough to be in the list. (Obviously)
# ### Variables vs Rank
#  - __The variables on the table has negligible to no correlation with the ranks.__ Therefore, rank does not favor any variable but rather, probably the combinations of those variables.
# 

# ## Working on the Data
# ### Importing Dependencies

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style='white')


# ## The Dataset
# **"Context:**
# 
# The top 50 most listened songs in the world by spotify. This dataset has several variables about the songs.
# 
# **Content:**
# 
# 50 songs 13 variables Data were stracted from: http://organizeyourmusic.playlistmachinery.com/ **"**
# 
# (source: https://www.kaggle.com/leonardopena/top50spotify2019)
# 
# The dataset consists of the following columns:
# - Rank
# - Track.Name - Name of the Track
# - Artist.Name - Name of the Artist
# - Genre - the genre of the track
# - Beats.Per.Minute - The tempo of the song.
# - Energy - The energy of a song - the higher the value, the more energtic. song
# - Danceability - The higher the value, the easier it is to dance to this song.
# - Loudness..dB.. - The higher the value, the louder the song.
# - Liveness - The higher the value, the more likely the song is a live recording.
# - Valence. - The higher the value, the more positive mood for the song.
# - Length. - The duration of the song.
# - Acousticness.. - The higher the value the more acoustic the song is.
# - Speechiness. - The higher the value the more spoken word the song contains.
# - Popularity - The higher the value the more popular the song is.

# In[ ]:


df = pd.read_csv('../input/top50spotify2019/top50.csv',encoding='latin-1')
df.rename(columns={'Unnamed: 0':'Rank'}, inplace=True)
df.head()


# ## Data Exploration: Visualization and Analysis

# As the variables cannot be generalized with the data consisting of only 50 samples, we will focus only on the song parameters and their compositions in the list, relationship of the variables with the rank, and not the relation of each variables.
# ### Composition of the List
# We will first look at what composes the top 50 songs.

# In[ ]:


plt.figure(figsize=(10,20))
sns.countplot(y='Artist.Name',data=df,order=df['Artist.Name'].value_counts().index).set(title='Number of Songs per Artist',
                                                                                       xlabel='Number of Songs',
                                                                                       ylabel='Artist')


# Among the top 50 songs, 4 of them belongs to Ed Sheeran. 2 more than the next runner-ups. So **the most listened artist is Ed Sheeran.**

# In[ ]:


plt.figure(figsize=(10,12))
sns.countplot(y='Genre',data=df,order=df['Genre'].value_counts().index).set(title='Number of Songs per Genre',
                                                                                       xlabel='Number of Songs',
                                                                                       ylabel='Genre')


# As it turns out, **Dance pop** is more pop than pop and it **is also the most listened genre.**

# In[ ]:


plt.figure(figsize=(15,5))
sns.distplot(df['Beats.Per.Minute']).set(title='Beats per Minute Distribution',
                                        xlabel='Beats per Minute',
                                        ylabel='Number of Songs (Normalized)')


# In[ ]:


print('BPM:')
print("Median:",df['Beats.Per.Minute'].median())
print("Standard Deviation:",df['Beats.Per.Minute'].std())
df['Beats.Per.Minute'].describe()


# Most songs runs from **96 to 137.5 BPM** which belongs to the spectrum of hiphop.(Source:https://www.izotope.com/en/learn/using-different-tempos-to-make-beats-for-different-genres.html) Which could mean most of the songs are of hip-hop-like BPM. The center BPM based on the median is 104.5 with standard deviation fo 30.898.

# In[ ]:


plt.figure(figsize=(10,15))
sns.countplot(y=df['Beats.Per.Minute'],order=df['Beats.Per.Minute'].value_counts().index).set(title='Beats per Minute Distribution',
                                        xlabel='Beats per Minute',
                                        ylabel='Number of Songs (Normalized)')


# Interestingly, when counts are plotted using bar plot, it can be seen that 176 is something special, having the most songs among other BPMs with 5 songs.

# In[ ]:


plt.figure(figsize=(15,5))
sns.distplot(df['Energy']).set(title='Energy Distribution',
                                        xlabel='Energy',
                                        ylabel='Number of Songs (Normalized)')


# In[ ]:


print('Energy:')
print("Median:",df['Energy'].median())
print("Standard Deviation:",df['Energy'].std())
df['Energy'].describe()


# This distribution plot generalizes that most of the songs in the list has a wide relative variance but with most of the values in the range of **55.25 to 74.75**. According to the distribution, the scale of this parameter might be between 0 to 100. If that's the case, then the most preferred songs are those with medium-high energy, 66.5 with standard deviation of 14.23. (This metric is according to Spotify)

# In[ ]:


plt.figure(figsize=(15,5))
sns.distplot(df['Danceability']).set(title='Danceability Distribution',
                                        xlabel='Danceability',
                                        ylabel='Number of Songs (Normalized)')


# In[ ]:


temp_param = 'Danceability'
print(temp_param)
print("Median:",df[temp_param].median())
print("Standard Deviation:",df[temp_param].std())
df[temp_param].describe()


# According to the graph, this parameter might be of scale 0 to 100. If so, then the listeners prefer medium-high from **67 to 79.75** danceability with median 73.5 and a sharp preference with standard deviation of 11.9298.

# In[ ]:


plt.figure(figsize=(15,5))
sns.distplot(df['Loudness..dB..']).set(title='Loudness Distribution',
                                        xlabel='Loudness',
                                        ylabel='Number of Songs (Normalized)')


# In[ ]:


temp_param = 'Loudness..dB..'
print(temp_param)
print("Median:",df[temp_param].median())
print("Standard Deviation:",df[temp_param].std())
df[temp_param].describe()


# The distribution of the loudness is distributed mostly at **-6.75 dB to -4.00 dB** centered at -6.0 dB, which, coincidentally, is at the medium-high part of the distribution as well.

# In[ ]:


plt.figure(figsize=(15,5))
sns.distplot(df['Liveness']).set(title='Liveness Distribution',
                                        xlabel='Liveness',
                                        ylabel='Number of Songs (Normalized)')


# Liveness is the likeness of the song being recorded live. it turns out, listeners prefer listening to studio versions of songs in the top 50.

# In[ ]:


plt.figure(figsize=(15,5))
sns.distplot(df['Length.']).set(title='Length Distribution',
                                        xlabel='Length (Minutes)',
                                        ylabel='Number of Songs (Normalized)')


# In[ ]:


temp_param = 'Length.'
print(temp_param)
print("Median:",df[temp_param].median())
print("Standard Deviation:",df[temp_param].std())
df[temp_param].describe()


# Most values are between **176.75 to 217.5 minutes** with 198 minutes being the central tendency. These are the values that determine what is too short and too long for a song.

# In[ ]:


plt.figure(figsize=(15,5))
sns.distplot(df['Valence.']).set(title='Valence Distribution',
                                        xlabel='Valence',
                                        ylabel='Number of Songs (Normalized)')


# In[ ]:


temp_param = 'Valence.'
print(temp_param)
print("Median:",df[temp_param].median())
print("Standard Deviation:",df[temp_param].std())
df[temp_param].describe()


# The data is surprisingly almost like a perfect bell curve with a little lean to the happier valence with a median of 55.5. The list is composed mostly in the balance of medium-high and medium-low happiness with the interquartile range in between **38.25 and 69.50**.

# In[ ]:


plt.figure(figsize=(15,5))
sns.distplot(df['Acousticness..'],kde=False).set(title='Acousticness Distribution',
                                        xlabel='Acousticness',
                                        ylabel='Number of Songs (Normalized)')


# In[ ]:


temp_param = 'Acousticness..'
print(temp_param)
print("Median:",df[temp_param].median())
print("Standard Deviation:",df[temp_param].std())
df[temp_param].describe()


# The interquartile range goes from **8.25 to 33.75 with mean at 15**, mostly at the lower part of the distribution. This means that out of the 50 most listened songs, only a few of them is acoustic.

# In[ ]:


plt.figure(figsize=(15,5))
sns.distplot(df['Speechiness.'],kde=False).set(title='Speechiness Distribution',
                                        xlabel='Speechiness',
                                        ylabel='Number of Songs (Normalized)')


# In[ ]:


temp_param = 'Speechiness.'
print(temp_param)
print("Median:",df[temp_param].median())
print("Standard Deviation:",df[temp_param].std())
df[temp_param].describe()


# With the interquartile range at **5 to 15**, it's clear that listeners are less likely to listen to songs with less words.

# In[ ]:


plt.figure(figsize=(15,5))
sns.distplot(df['Popularity']).set(title='Popularity Distribution',
                                        xlabel='Popularity',
                                        ylabel='Number of Songs (Normalized)')


# In[ ]:


temp_param = 'Popularity'
print(temp_param)
print("Median:",df[temp_param].median())
print("Standard Deviation:",df[temp_param].std())
df[temp_param].describe()


# Most of the songs' popularity in the list are between **86 and 90.75**. This means that undoubtly, popular songs are mostly listened to. Even the minimum value is relatively high at 70. The reason why it is not higher might be because of songs that are comedic in nature which are known so much but are not frequently listened to.

# ## Rank vs Variables
# Here, we'll look at the correlation of the variables according to rank.

# In[ ]:


rank_corr = df.corr()['Rank'].drop(index='Rank').sort_values(ascending=False)
print(rank_corr)


# By correlation, we can see that there are little to no correlations (<0.30) between the rank and the other parameters. In case there is possibly a polynomial/parabolic relation, we will have to visualize them.

# In[ ]:


sns.pairplot(df,
             x_vars='Rank',
             y_vars=df[rank_corr.index].columns,
             kind='reg',
             aspect=7)


# Looking the plots alone, everything seems to be just a random scatter plot with regressions diverging so much. Therefore, there is really little to no correlation between the rank and the other variables.
