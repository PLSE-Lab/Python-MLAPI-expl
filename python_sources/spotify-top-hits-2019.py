#!/usr/bin/env python
# coding: utf-8

# # Top Hits of 2019 (Spotify)
# ***
# ### Table Column Info:
# 
# 1. Track.Name: Name of the Track
# 2. Artist.Name:Name of the Artist
# 3. Genre: the genre of the track
# 4. Beats.Per.Minute: The tempo of the song.
# 5. Energy: The energy of a song - the higher the value, the more energtic song
# 6. Danceability: The higher the value, the easier it is to dance to this song.
# 7. Loudness..dB..: The higher the value, the louder the song.
# 8. Liveness: The higher the value, the more likely the song is a live recording.
# 9. Valence. : The higher the value, the more positive mood for the song.
# 10. Length. : The duration of the song.
# 11. Acousticness.. : The higher the value the more acoustic the song is.
# 12. Speechiness. : The higher the value the more spoken word the song contains.
# 13. Popularity :The higher the value the more popular the song is.
# 14. Lyrics: Original Lyrics of the Song if available. 
# 
# This dataset orignated from https://www.kaggle.com/leonardopena/top50spotify2019 with lyrics was added to the original dataset. Unhash the codes below and run top50.csv to generate lyrics.  
# 
# 
# ### Question:
# - What are the features that contribute to song popularity?
# - Predict song popularity?
# ---

# # 1. Import Modules 

# In[ ]:


get_ipython().system('pip install lyrics_extractor')


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from lyrics_extractor import Song_Lyrics
import nltk
from nltk.corpus import stopwords
from textblob import TextBlob
from collections import Counter


# In[ ]:


import cufflinks as cf
import chart_studio.plotly

from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go

init_notebook_mode(connected=True)
cf.go_offline()


# In[ ]:


sns.set_style('darkgrid')
sns.set_color_codes("pastel")


# In[ ]:


spotify = pd.read_csv("../input/spotify-top-50-w-lyrics/spotify_w_lyrics.csv",  encoding = "ISO-8859-1" , index_col= 0)


# In[ ]:


spotify.head(10)


# # 2. Data Cleaning

# In[ ]:


spotify.columns = [cols.replace('.', '') for cols in spotify.columns]
spotify = spotify.sort_values(by = 'Popularity', ascending =False).reset_index(drop = True)


# In[ ]:


spotify.describe()


# In[ ]:


spotify.info()


# ---
# # 3. EDA
# ### 3a. Numerical Data Analysis

# In[ ]:


# target variable: popularity 
sns.distplot(spotify['Popularity'], kde = False, bins = 10)


# - top 50 songs have popularity score of around 87.5 - 92.5.

# In[ ]:


# distribution of other features

x_cols = ['BeatsPerMinute', 'Energy','Danceability', 'LoudnessdB', 'Liveness', 'Valence', 'Length','Acousticness', 
          'Speechiness']

fig, axes = plt.subplots(nrows=3, ncols=3, figsize = (15,15))
for i, x_col in enumerate(x_cols):
    sns.distplot( spotify[x_col], ax=axes[i//3,i%3], kde = False, bins = 10)
    axes[i//3,i%3].set_xlabel(x_col) 


# In[ ]:


# How does features relate to popularity

x_cols = ['BeatsPerMinute', 'Energy','Danceability', 'LoudnessdB', 'Liveness', 'Valence', 'Length','Acousticness', 
          'Speechiness']

fig, axes = plt.subplots(nrows=3, ncols=3, figsize = (15,15))
for i, x_col in enumerate(x_cols):
    sns.regplot( x = x_col,  y = 'Popularity', ax=axes[i//3,i%3], data = spotify)
    axes[i//3,i%3].set_xlabel(x_col) 


# - Popular songs tend to have higher `BeatsPerMinute`, `speechiness` and **lower** `valence`.

# In[ ]:


# correlation heatmap between X features and popularity

fig = plt.figure(figsize = (10,8))

mask = np.zeros_like(spotify.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(spotify.corr(), annot = True,linewidths = 0.3, mask = mask)


# - `BeatsPerMinute` co-related to `Speechiness`, this may introduce problems of colinearity in regression later. 
# - `Energy` co-related to `LoudnessB`, `Valence` (happiness!).

# ### 3b. Categorical Analysis

# In[ ]:


# Artist Popularity:
fig = plt.figure(figsize = (15,8))
sns.set_style('whitegrid')
artist = spotify.groupby('ArtistName').size().reset_index(name = 'count')
artist = artist.sort_values(by = 'count', ascending =False)
sns.barplot(y = 'ArtistName',x="count", data=artist,  color="b")
sns.despine(left=True, bottom=True)


# - Ed Sheeren is one of the most popular artist in 2019! Unforuntely, categorisation by artist names cant tell us much sinec most artist only has one popular songs of 2019. 

# In[ ]:


# Genre Popularity:

spotify['Genre'].unique()


# - Similar to the categorisation of artistname there are too many sub genres that makes the classification a little meaningless. The next section groups these genres into parent genres and sub genres from other langs are classified under others. Categorisation can be a little subjective since there is no clear distinction for some `Genres`.

# In[ ]:


def parent_genre(genre):
    music_genre = {'electronic' : ['electropop','trap music','pop house', 'big room', 'brostep' ,'edm'],
                   'hip hop/rap': ['canadian hip hop','atl hip hop','reggaeton','reggaeton flow','dfw rap',
                                   'country rap'],
                   'pop': ['pop','panamanian pop', 'canadian pop', 'australian pop', 'dance pop', 'boy band'],
                   'others': ['escape room', 'latin','r&b en espanol']}
    
    for parent, sub in music_genre.items():
        if genre in sub:
            return parent


# In[ ]:


spotify['parent_genre'] = spotify['Genre'].apply(parent_genre)


# In[ ]:


# Genre Popularity:
import plotly.graph_objects as go

artist = spotify.groupby('parent_genre').size().reset_index(name = 'count')
artist = artist.sort_values(by = 'count', ascending =False)

values = artist['count'].tolist()
labels = artist['parent_genre'].tolist()

fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
fig.update_traces(textposition='inside', textinfo='value+label', title_text = 'Parent Genre Segmentation')
fig.show()


# In[ ]:


parent_genre = pd.get_dummies(spotify['parent_genre'],drop_first=True)


# In[ ]:


spotify = pd.concat([spotify, parent_genre],axis=1)


# In[ ]:


spotify.head()


# - pop songs seems to be popular amongst listeners with dance pop being the most popular followed by hip/hop. Dummy variables are then created to differnetiate between the parent genres.
# - The improportionate ratios across categories could be attributed to the number of sub genres within each category which then skews the parent genre.

# ### 3c. Feature Engineering: Lyrical Analysis
# Lyrics from lyrics-extractor library; following the instructions in pypi page to generate GCS_API_KEY, GCS_ENGINE_ID tokens. Unfortuntely unable to run lyrics_extractor py on Kaggle and have attached the data set with lyrics in this kernel. The original code is attached below for reference.
# 
# For more info, visit https://pypi.org/project/lyrics-extractor/

# In[ ]:


# Original function that scrap for song lyrics
# def get_lyrics(track):
#     extract_lyrics = Song_Lyrics('GCS_API_KEY', 'GCS_ENGINE_ID')
#     song_title, song_lyrics = extract_lyrics.get_lyrics(track)
#     return song_lyrics


# In[ ]:


# spotify['lyrics'] = spotify['TrackName'].apply(lambda row: (get_lyrics(row)))
# spotify = spotify.replace('', 'None')


# In[ ]:


def pre_processing(lyrics):
    lyrics = lyrics.replace('\n', ' ',).lower()
    lyrics = re.sub(r'\[.*?\]', '', lyrics)
    lyrics = re.sub(r'\(.*?\)', '', lyrics)
    lyrics = re.sub(r'\{.*?\}', '', lyrics)
    
    lyrics = re.sub(r'[^a-zA-Z0-9 ]', '', lyrics)
    lyrics = ' '.join(lyrics.split())
    return lyrics

def count_unique(df):
    text = df['Lyrics']
    stop_words = stopwords.words('english')
    newStopWords = ['youre','im', 'ill','ive', 'm', 'oh' , 'yeh', 'yeah', 'dont', 'got', 'gonna', 'wanna']
    stop_words.extend(newStopWords)
    stopwords_dict = Counter(stop_words)
    
    initial_len = len(text.split())
    clean_lyrics = ' '.join([word for word in text.split() if word not in stopwords_dict])
    text = set([word for word in text.split() if word not in stopwords_dict])
    unique_length = (len(text)/initial_len)*100
    return clean_lyrics, unique_length


# In[ ]:


spotify['Lyrics'] = spotify['lyrics'].apply(lambda lyrics:pre_processing(lyrics))
spotify['text_length'] = spotify['Lyrics'].apply(lambda lyrics: len(lyrics.split()))
spotify[['clean_lyrics','unique_length']] = spotify.apply(count_unique, result_type='expand', axis = 1)


# In[ ]:


spotify.head()


# In[ ]:


def sentiment_analysis(lyrics):
#   TextBlob has a function that allows for translation of text to eng, 
#   its still possible to run sentiment analysis even without translating the lyrics.
    blob = TextBlob(lyrics)
    language = blob.detect_language()
    if language != 'en':
        blob = blob.translate(to="en")

    for sentence in blob.sentences:
        sentiment = sentence.sentiment.polarity
    return sentiment


# In[ ]:


spotify['sentiment'] = spotify['clean_lyrics'].apply(sentiment_analysis)


# In[ ]:


lexicalrichness = spotify[spotify['text_length'] > 1]


# In[ ]:


fig, axes = plt.subplots(nrows=1, ncols=3, figsize = (18,5))
sns.set_style('darkgrid')
cols = ['text_length', 'unique_length', 'sentiment']
 
for i, x_col in enumerate(cols):
    sns.distplot(lexicalrichness[x_col], ax=axes[i], kde = False, bins = 15)
    axes[i].set_xlabel(x_col) 


# In[ ]:


fig, axes = plt.subplots(nrows=1, ncols=3, figsize = (18,5))
sns.set_style('darkgrid')
cols = ['text_length', 'unique_length', 'sentiment']
 
for i, x_col in enumerate(cols):
    sns.scatterplot(x = x_col, y = 'Popularity' , data = lexicalrichness, ax=axes[i])
    axes[i].set_xlabel(x_col) 


# - Most songs have text length of **~400**
# - Most songs have unique length (also known as lexical richness, measure of normalised unique words) of **20 & 30%.**
# - Sentiment and popularity doesnt seem to be corelated, most songs have no polarity (0). Contrary to the previous findings, songs with higher sentiment (positivity) tends to be more popular here.
# 
# $$ unique \ length = \frac {num\ of \ unique \ words}{length \ of \ words \ excluding \ nltk \ stopwords}$$

# In[ ]:


# wordcloud for most popular words 
# Here's a word cloud for those curious on popular words.
from wordcloud import WordCloud
def show_wordcloud(data, title = None):
    wordcloud = WordCloud(
        background_color='white',
        max_words=200,
        max_font_size=40, 
        scale=3,
        random_state=1 # chosen at random by flipping a coin; it was heads
    ).generate(str(data))

    fig = plt.figure(1, figsize=(12, 12))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)
    plt.show()

show_wordcloud(spotify['clean_lyrics'])


# # 4. Linear Regression for Song Popularity

# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn import metrics
import statsmodels.api as sm
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms


# ### 4a. Modelling

# In[ ]:


spotify.columns


# In[ ]:


# we shall skip observation withtext length = 1 as it may skew our results. 
df = spotify[spotify['text_length'] > 1]


# In[ ]:


"""
To avoid the problems associated with co-linearity, only BeatsPerMinute is used 
as the dependent variable between the two. 
"""

y = df['Popularity']
X = df[['BeatsPerMinute','Valence','unique_length', 'hip hop/rap', 'others', 'pop']]


# In[ ]:


lm = LinearRegression()
lm.fit(X,y)
predictions = lm.predict(X)


# In[ ]:


X2 = sm.add_constant(X)
est = sm.OLS(y, X2)
est2 = est.fit()
print(est2.summary())


# **Statiscal Analysis**
# - R2 = 0.480, dependent variables does not explain the y variable well. 
# - Discrete variables like `pop`, `Valence`, `unique_length` are statisically signficant (<0.05) and that a non-zero corelation exists. 

# ### 4b. Predictions vs Actual

# In[ ]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (15,5))
sns.scatterplot(x = y,y = predictions, ax = ax1)
line = mlines.Line2D([0, 1], [0, 1], color='red')
transform = ax1.transAxes
line.set_transform(transform)
ax1.add_line(line)
ax1.set_xlim([70,100])
ax1.set_ylim([70,100])

sns.distplot((y-predictions),bins=10, ax= ax2);


# In[ ]:


MAE = metrics.mean_absolute_error(y, predictions)
MSE = metrics.mean_squared_error(y, predictions)
RMSE = np.sqrt(metrics.mean_squared_error(y, predictions))

error_df = pd.DataFrame(data = [MAE, MSE, RMSE], index = ['MAE', 'MSE', "RMSE"], columns=['Error'])
error_df


# - Linear regression model definitely unable to perform well given poor data set and for this data set, we assume that all test = train data since there are too little datapoints to sufficiently create meaningful train/test set. Thus, the linear regression model will definitely be overfitted with current data. 
# - Song popularity is also predicted to be slightly greater than the actual value.

# ---
# ## Concluding Remarks:
# - Receipe for popular songs? Try creating a pop song that more unique words and is mildly more melancholic! 
# - While some important features were identified, the data set is rather small and the results should be taken with a pinch of salt. Feel free to leave any comments/ suggestions below , hope you enjoyed the kernel! 
