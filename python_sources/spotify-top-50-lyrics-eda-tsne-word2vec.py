#!/usr/bin/env python
# coding: utf-8

# **Loading of libraries**

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import chardet
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from pylab import rcParams
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import Word2Vec
from sklearn.manifold import TSNE


# # Loading of Datasets

# In[ ]:


with open('/kaggle/input/top50spotify2019/top50.csv', 'rb') as f:
    result = chardet.detect(f.read())
    

songs_df = pd.read_csv("../input/top50spotify2019/top50.csv", encoding=result['encoding'])
lyrics_df = pd.read_csv("../input/songs-lyrics/Lyrics.csv")


# In[ ]:


songs_df.shape


# In[ ]:


songs_df.head()


# **Dropping Unamed column since it has no significance**

# In[ ]:


songs_df = songs_df.drop('Unnamed: 0', axis = 1 )


# In[ ]:


songs_df.info()


# In[ ]:


lyrics_df.info()


# In[ ]:


lyrics_df.shape


# In[ ]:


lyrics_df.head()


# In[ ]:


songs_df.describe()


# # Visualizing Dataset

# In[ ]:


songs_categorical_cols = ['Track.Name','Artist.Name','Genre']


# In[ ]:


songs_df.describe()


# In[ ]:


rcParams['figure.figsize'] = 10, 20
songs_df.drop(songs_categorical_cols,axis=1).hist();


# Length and popularity almost follow a normal distribution.

# In[ ]:


rcParams['figure.figsize'] = 8, 5
sns.heatmap(songs_df.drop(songs_categorical_cols,axis=1).corr());


# In[ ]:


sns.pairplot(songs_df.drop(songs_categorical_cols,axis=1));


# # Visualizing Individual Attributes

# ## Numerical Attributes

# In[ ]:


## Popularity
sns.boxplot( y = songs_df["Popularity"]);


# Most of the songs in the top 50, have a popularity around 85, presumably there are very small outliers compared to the median.

# In[ ]:


fig, ax = plt.subplots(1,3)
fig.subplots_adjust(hspace=0.6, wspace=0.6)

sns.boxplot( y = songs_df["Beats.Per.Minute"], ax=ax[0])
sns.boxplot( y = songs_df["Energy"], ax=ax[1])
sns.boxplot( y = songs_df["Danceability"], ax=ax[2])

fig.show()


# Most of the songs of the top 50, possess a dancebility around 70, presumably there are very small outliers values compared to the median.

# In[ ]:


fig, ax = plt.subplots(1,3)
fig.subplots_adjust(hspace=0.6, wspace=0.6)

sns.boxplot( y = songs_df["Loudness..dB.."], ax=ax[0])
sns.boxplot( y = songs_df["Liveness"], ax=ax[1])
sns.boxplot( y = songs_df["Valence."], ax=ax[2])

fig.show()


# In[ ]:


fig, ax = plt.subplots(1,3)
fig.subplots_adjust(hspace=0.8, wspace=0.8)

sns.boxplot( y = songs_df["Length."], ax=ax[0])
sns.boxplot( y = songs_df["Acousticness.."], ax=ax[1])
sns.boxplot( y = songs_df["Speechiness."], ax=ax[2])

fig.show()


# Most of the songs of the top 50, possess a loudness value with around -6db, however there are atypical values close to -10db.
# 
# Same is the case of liveness, where most songs have low values, where only those with high values are songs recorded live.

# In[ ]:


rcParams['figure.figsize'] = 10, 8
wordcloud = WordCloud(
    width = 3000,
    height = 2000,
    background_color = 'black').generate(str(songs_df.Genre.values))

plt.imshow(wordcloud, interpolation = 'bilinear');


# The largest number of songs in the top 50 correspond to pop genre songs

# In[ ]:


rcParams['figure.figsize'] = 10, 8
wordcloud = WordCloud(
    width = 3000,
    height = 2000,
    background_color = 'black').generate(str(songs_df['Artist.Name'].values))

plt.imshow(wordcloud, interpolation = 'bilinear');


# ## Variable Interactions

# 
# There is a relationship between Popularity and the following variables:
# 
# * More Danceability More Popularity?
# * More Acousticness More Popularity?
# * More Speechiness More Popularity?
# * Less Liveness More Popularity?
# * Most popular Genre?
# 

# In[ ]:


dataset = pd.merge(songs_df, lyrics_df, left_on='Track.Name', right_on='Track.Name')
dataset['Lyrics'] = dataset['Lyrics'].astype(str)


# In[ ]:


sns.jointplot(x='Danceability', y='Popularity', 
              data=dataset, kind='scatter');


# seems like danceability plays an important role in song's popularity

# In[ ]:


sns.jointplot(x='Acousticness..', y='Popularity', 
              data=dataset, kind='scatter');


# Acousticness is inversely proportional to Popularity

# In[ ]:


sns.jointplot(x='Loudness..dB..', y='Popularity', 
              data=dataset, kind='scatter');


# there is no direct relationship between Loudness and Popularity

# In[ ]:


sns.jointplot(x='Liveness', y='Popularity', 
              data=dataset, kind='scatter');


# Liveness is distributed between ~ (1-20) which means the less lively the song is the more popular it is. 

# **Genres by Popularity**

# In[ ]:


rcParams['figure.figsize'] = 15, 8
popular_genre = dataset[['Genre','Popularity']].groupby('Genre').sum().sort_values(ascending=False,by='Popularity')[:10]

sns.barplot(x=popular_genre.index, y=popular_genre['Popularity'], data=popular_genre, palette = "pastel");


# # Word2Vec on Lyrics

# In[ ]:


dataset['Lyrics'] = dataset['Lyrics'].str.lower().replace(r'\n',' ')


# In[ ]:


stop_words_en = list(stopwords.words("english"))
stop_words_es = list(stopwords.words("spanish"))
punctuations = list(string.punctuation)
forbidden = ['(',')',"'",',','oh',"'s", 'yo',"'ll", 'el', "'re","'m","oh-oh","'d", "n't", "``", "ooh", "uah", "'em", "'ve", "eh", "pa", "brr", "yeah"] 
stop_words_all = set(stop_words_en + stop_words_es + punctuations + forbidden)


# In[ ]:


def cleanse_text(tokens):
    return [i for i in tokens if ((i not in list(stop_words_all)) and (re.search(r'\d+', i) == None)) ]


# In[ ]:


songs = []
for song in dataset.Lyrics.values:
    songs.append(cleanse_text(word_tokenize(song)))


# In[ ]:


word2vec = Word2Vec(songs, min_count=5)


# # TSNE to visualize Word2Vec

# In[ ]:


def tsne_plot(model):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []

    for word in model.wv.vocab:
        tokens.append(model[word])
        labels.append(word)
    
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    plt.figure(figsize=(16, 16)) 
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show();


# In[ ]:


tsne_plot(word2vec)

