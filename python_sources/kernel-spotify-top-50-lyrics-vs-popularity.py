#!/usr/bin/env python
# coding: utf-8

# **Loading of libraries**

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import chardet
import nltk
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk.corpus import sentiwordnet as swn
import string


# **Loading of Datasets**
# * raw_data -> dataset about the top 50 songs
# * raw_data_lyrics -> dataset about the lyrics of the songs

# In[ ]:


with open('/kaggle/input/top50spotify2019/top50.csv', 'rb') as f:
    result = chardet.detect(f.read())
    

raw_data = pd.read_csv("../input/top50spotify2019/top50.csv", encoding=result['encoding'])
raw_data_lyrics = pd.read_csv("../input/songs-lyrics/Lyrics.csv")


# In[ ]:


raw_data.info()


# In[ ]:


raw_data = raw_data.drop('Unnamed: 0', axis = 1 )


# I decide to remove the column "Unnamed: 0", because it is not significant for the model.

# In[ ]:


raw_data.shape


# In[ ]:


raw_data.head(3)


# In[ ]:


raw_data_lyrics.info()


# In[ ]:


raw_data_lyrics.head(3)


# In[ ]:


raw_data_lyrics.shape


# In[ ]:


raw_data.columns = ['Track_Name','Artist_Name','Genre','BPM','Energy','Danceability','Loudness', 'Liveness', 'Valence','Length', 'Acousticness', 'Speechiness','Popularity']


# In[ ]:


raw_data.describe()


# **Target: Popularity**

# In[ ]:


sns.boxplot( y = raw_data["Popularity"])


# Most of the songs in the top 50, have a popularity around 85, presumably there are very small outliers compared to the median.

# In[ ]:


fig, ax = plt.subplots(1,3)
fig.subplots_adjust(hspace=0.6, wspace=0.6)

sns.boxplot( y = raw_data["BPM"], ax=ax[0])
sns.boxplot( y = raw_data["Energy"], ax=ax[1])
sns.boxplot( y = raw_data["Danceability"], ax=ax[2])

fig.show()


# Most of the songs of the top 50, possess a dancebility around 70, presumably there are very small outliers values compared to the median.

# In[ ]:


fig, ax = plt.subplots(1,3)
fig.subplots_adjust(hspace=0.6, wspace=0.6)

sns.boxplot( y = raw_data["Loudness"], ax=ax[0])
sns.boxplot( y = raw_data["Liveness"], ax=ax[1])
sns.boxplot( y = raw_data["Valence"], ax=ax[2])

fig.show()


# In[ ]:


fig, ax = plt.subplots(1,3)
fig.subplots_adjust(hspace=0.8, wspace=0.8)

sns.boxplot( y = raw_data["Length"], ax=ax[0])
sns.boxplot( y = raw_data["Acousticness"], ax=ax[1])
sns.boxplot( y = raw_data["Speechiness"], ax=ax[2])

fig.show()


# Most of the songs of the top 50, possess a loudness value with around -6db, however there are atypical values close to -10db.
# 
# Same is the case of liveness, where most songs have low values, where only those with high values are songs recorded live.

# In[ ]:


sns.catplot(y = "Genre", kind = "count",
            palette = "pastel", edgecolor = ".6",
            data = raw_data)


# The largest number of songs in the top 50 correspond to pop genre songs

# In[ ]:


sns.catplot(x = "Popularity", y = "Genre", kind = "bar" ,
            palette = "pastel", edgecolor = ".6",
            data = raw_data)


# However, the pop genre does not determine the popularity of the songs since there are songs from other genres with higher values of popularity.

# In[ ]:


sns.pairplot(raw_data)


# Let's take a closer look at the "Popularity" graphics.  There doesn't seem to be a strong relationship between popularity and BPM, Energy, Loudness, Valence, Length.
# 
# However, there is a relationship between Popularity and the following variables:
# 
# * More Danceability More Popularity?
# * More Acousticness More Popularity?
# * More Speechiness More Popularity?
# * Less Liveness More Popularity?
# 

# In[ ]:


dataset = pd.merge(raw_data, raw_data_lyrics, left_on='Track_Name', right_on='Track.Name')
del dataset['Track.Name']
dataset['Lyrics'] = dataset['Lyrics'].astype(str)


# In[ ]:


dataset.head(3)


# In[ ]:


dataset = dataset.drop(dataset.index[[30, 22]])


# I eliminate 2 registers with problems in the name of the song and with repeated songs.

# In[ ]:


dataset['Lyrics'] = dataset['Lyrics'].str.lower().replace(r'\n',' ')


# In[ ]:


tokens = dataset['Lyrics'].fillna("").map(nltk.word_tokenize)


# In[ ]:


stop_words_en = set(stopwords.words("english"))
stop_words_es = set(stopwords.words("spanish"))


# In[ ]:


punctuations = list(string.punctuation)


# In[ ]:


forbidden = ['(',')',"'",',','oh',"'s", 'yo',"'ll", 'el', "'re","'m","oh-oh","'d", "n't", "``", "ooh", "uah", "'em", "'ve", "eh", "pa", "brr", "yeah"] 


# Sentiment?

# In[ ]:


tokens = [i for i in tokens if i not in list(stop_words_en)]
tokens = [i for i in tokens if i not in list(stop_words_es)]
tokens = [i for i in tokens if i not in forbidden]

tokens_= []
for w in tokens:
    g = []
    for strg in w:
        if (strg not in forbidden):
            g.append(strg)
    tokens_.append(g)


# **I got stuck, any ideas?**
