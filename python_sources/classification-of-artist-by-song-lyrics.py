#!/usr/bin/env python
# coding: utf-8

# # Artist classification by song lyrics
# 
# 2 Nov 2019  
# Reinhard Sellmair
# 
# 
# 1. [Introduction](#Introduction)
# 2. [Pre-processing](#Pre-processing)  
#     2.1 [Brackets](#Brackets)  
#     2.2 [Line breaks](#Line_breaks)   
#     2.3 [Remove non-english songs](#Remove_non-english_songs)  
#     2.4 [Tokenization](#Tokenization)  
#     2.5 [Stemming](#Stemming)  
# 3. [EDA](#EDA)  
#     3.1 [Number of songs, artists, and words](#Number_of_songs_artists_and_words)  
#     3.2 [Duplicated songs](#Duplicated_songs)  
# 4. [Feature engineering](#Feature_engineering)  
#     4.1 [Number of words](#Number_of_words)  
#     4.2 [Repeated words](#Repeated_words)  
#     4.3 [Words per line](#Words_per_line)  
#     4.4 [Word frequency (TFIDF)](#Word_frequency_TFIDF)  
#     4.5 [Sentiment analysis](#Sentiment_analysis)  
# 5. [Prediction](#Prediction)  
# 6. [Validation](#Validation)  
# 7. [Potential improvements](#Potential_improvements)  

# <a id="Introduction"></a>
# ## Introduction

# In this kernel I develop a machine learning approach to match artists (singer or band) to their song lyrics. 
# 
# This is very challanging as there are so many artist while the number of songs per artist is relatively small. Thus, there is probably not enough data to train a multi-class classifier where each class represnts an artist. Furthermore, this approach would be very inflexible as it would require to train the whole model again when just adding one artist. 
# 
# Instead, I first match the song lyrics with each artist and predict for each pair separately if this song belongs to this artist. For example if I have one song s1 and a set of three artist a1, a2, a3, I create pairs s1 - a1, s1 - a2, and s1 - a3 and get prediction probabilities for each pair: p1, p2, p3. Then I select the artist - song pair with the highest probability. 
# 
# To make these predictions I extract features for each song and each artist. 
# 
# Next in this kernel I will pre-process the data, do an exploratory data analysis (EDA), engineer features, build and train a machine learning model and finally validate the model's accuracy.

# <a id="Pre-porcessing"></a>
# ## Pre-processing

# In this part I have the first look at the data, remove unnecessary information and transform it so that I can do the EDA next.
# 

# In[ ]:


# import libraries
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import re
import random
import nltk
from scipy import sparse
from scipy.sparse import csr_matrix, vstack
from textblob import TextBlob
from langdetect import detect_langs
import pickle
from datetime import datetime

from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# In[ ]:


# import data
song_df = pd.read_csv('../input/songlyrics/songdata.csv')
song_df.head()


# The dataset contains four columns:
# - artist: name of artist
# - song: name of song
# - link: song reference (http://www.lyricsfreak.com needs to be added in front)
# - text: text of song lyric
# 
# All columns except the `link` are relevant for my work.
# The `artist` and `song` columns are fine already, let's have a look at some `text` entries:

# In[ ]:


print(song_df['text'].iloc[7000])


# In[ ]:


print(song_df['text'].iloc[10000])


# <a id="Brackets"></a>
# ### Brackets

# It looks like the texts are formated in different styles: the first example includes text in round brackets which says how often parts of the text shall be repeated e.g. "(repeat previous line 3 times)". The second example does not have this but describes the type of lyrics in square brackets like "[Chorus]".
# 
# First I have a look at round brackets:

# In[ ]:


text_in_round_brackets = sum(list(song_df['text'].map(lambda s: re.findall(r'\((.*?)\)',s))), [])
print('Number of round brackets: {}'.format(len(text_in_round_brackets)))


# In[ ]:


random.seed(0)
random.choices(text_in_round_brackets, k=20)


# In total there are 63 thousand round brackets. Above are 20 randomly chosen content within brackets. Most of these seem to be part of the songs' text.
# 
# Next I search for square brackets.

# In[ ]:


text_in_square_brackets = sum(list(song_df['text'].map(lambda s: re.findall(r'\[(.*?)\]',s))), [])
print('Number of square brackets: {}'.format(len(text_in_square_brackets)))


# In[ ]:


random.seed(0)
random.choices(text_in_square_brackets, k=20)


# There are more than 29 thousand square brackets. Looks like the text in square brackets does mostly not belong to the lyrics of the song and should be removed.
# 
# Let's check curly brackets:

# In[ ]:


text_in_curly_brackets = sum(list(song_df['text'].map(lambda s: re.findall(r'\{(.*?)\}',s))), [])
print('Number of square brackets: {}'.format(len(text_in_curly_brackets)))


# There are no curly brackets at all.
# 
# Since the text in round brackets seems to belong to the song lyrics I only remove the brackets and keep the content. As square brackets usually do not contain text belonging to the lyrics I remove them including their content:

# In[ ]:


# remove round brackets but not text within
song_df['text'] = song_df['text'].map(lambda s: re.sub(r'\(|\)', '', s))

# remove square brackest and text within
song_df['text'] = song_df['text'].map(lambda s: re.sub(r'\[(.*?)\] ', '', s))


# <a id="Line_breaks"></a>
# ### Line breaks

# The texts also contains `\n` which indicate line breaks. As this is also does not belong to the lyrics I remove all line breaks. However, the number of lines or number of words per line could vary from artist to artist I add the number of lines as new column before removing the breaks.

# In[ ]:


# count number of lines
song_df['lines'] = song_df['text'].map(lambda t: len(re.findall(r'\n', t)))
# remove line breaks
song_df['text'] = song_df['text'].map(lambda s: re.sub(r' \n|\n', '', s))


# <a id="Remove_non-english_songs"></a>
# ### Remove non-english songs

# Not all songs are in english language. As I will later do some text analysis which is language specific I remove all non-english songs. To do that I use `detect_langs` to get the probability that a text is in english language.

# In[ ]:


def get_eng_prob(text):
    detections = detect_langs(text)
    for detection in detections:
        if detection.lang == 'en':
            return detection.prob
    return 0

song_df['en_prob'] = song_df['text'].map(get_eng_prob)

print('Number of english songs: {}'.format(sum(song_df['en_prob'] >= 0.5)))
print('Number of non-english songs: {}'.format(sum(song_df['en_prob'] < 0.5)))


# There are 470 songs which probability of being english is less than 50%. I remove all those songs from the data.

# In[ ]:


song_df = song_df.loc[song_df['en_prob'] >= 0.5]


# <a id="Tokenization"></a>
# ### Tokenization

# A very common way to analyse text is to seperate it into a list of words which makes it much easier to do further analysis. I'm using `nltk.tokenize` to do that. Furthermore, all punctuations are removed as well. Below is an example how the text is converted.

# In[ ]:


tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
song_df['tokens'] = song_df['text'].map(tokenizer.tokenize)

print('Text:')
print(song_df['text'].iloc[0])

print('Tokens:')
print(song_df['tokens'].iloc[0])


# <a id="Stemming"></a>
# ### Stemming

# Stemming is a simple method to convert words to a common base form. For example converting plural to singular or past tense to present tense. This helps to treat words with the same meaning in the same way.
# 
# In contrast to lemmatization it uses a very simple method and does not always find the grammatically correct from. However, as stemming is much faster than lemmatization I'm using it here. 
# 
# Below is an example how different forms of the word "make" are converted. In this case stemming failes to correctly convert the past tense.

# In[ ]:


# initialise stemmer
stemmer = nltk.stem.porter.PorterStemmer()

token = 'make'
print('{} -> {}'.format(token, stemmer.stem(token)))

token = 'makes'
print('{} -> {}'.format(token, stemmer.stem(token)))

token = 'making'
print('{} -> {}'.format(token, stemmer.stem(token)))

token = 'made'
print('{} -> {}'.format(token, stemmer.stem(token)))


# Below I stem all tokens.

# In[ ]:


# create dictionary to map tokens their stem
token_to_stem = {}
# initialise word count
token_count = 0
# iterate through all songs
for lst in song_df['tokens']:
    # iterate through all tokens of song
    for token in lst:
        token_count += 1
        # check if token is in dictionary
        if token not in token_to_stem:
            # add token to dictionary
            token_to_stem[token] = stemmer.stem(token)
            
song_df['stems'] = song_df['tokens'].map(lambda lst: [token_to_stem[token] for token in lst])

print('Number of tokens: {}'.format(token_count))
print('Number of unique tokens: {}'.format(len(token_to_stem.keys())))
print('Number of unique stems: {}'.format(len(set(token_to_stem.values()))))


# Stemming reduced the number of unique tokens from 103 thousand to 57 thousand.

# <a id="EDA"></a>
# ## EDA

# In this part I do an exploratory data analysis to get more familiar with the dataset.

# <a id="Number_of_songs_artists_and_words"></a>
# ### Number of songs, artists, and words

# In[ ]:


# number of songs
print('number of songs: ', str(len(song_df)))

# number of artists
print('number of artists: ', str(len(song_df['artist'].unique())))

# distribution songs per artist
song_count_df = song_df.groupby('artist')[['song']].count()
fig = px.histogram(song_count_df, x='song', title='Songs per artist', labels={'song': 'Songs'})
fig.show()


# There are 57 thousand songs from 638 artists, on average there are 90 songs per artist. 
# 
# The histogrm shows the distribution of songs per artist. We can see that there is quite a wide range: 28 artists have less than 10 songs while 22 have even more than 180 songs. 
# 
# These differences may cause different accuracies for matching songs to the correct artist - artists with very few songs will be much more difficult to match.

# In[ ]:


# words per song
song_df['n_stems'] = song_df['stems'].map(len)

fig = px.histogram(song_df, x='n_stems', title='Words per song')
fig.show()


# The histogram shows the distribution of number of words per song. Typically songs have 150 to 250 words. The distribution has a long tail - there are songs with far more words (up to 900).

# <a id="Duplicated_songs"></a>
# ### Duplicated songs

# Songs can be covered by other artists, so it is possible that there are duplicated songs from different artists in the dataset.

# In[ ]:


# create dataframe with lists of artists
song_df['stems_str'] = song_df['stems'].map(lambda lst: ' '.join(lst))

# map text to artists
stems_to_artist = {}
for tp in song_df[['artist', 'stems_str']].itertuples(index=False):
    artist = tp[0]
    stems = tp[1]
    if stems in stems_to_artist:
        stems_to_artist[stems].append(artist)
    else:
        stems_to_artist[stems] = [artist]

# insert list of artists to dataframe
song_df['artists'] = song_df['stems_str'].map(stems_to_artist)
song_df['duplicates'] = song_df['artists'].map(len) - 1

# convert list of artists to set of artists
song_df['artists'] = song_df['artists'].map(set)
song_df['n_artists'] = song_df['artists'].map(len)

# remove duplicate songs
artist_text_df = song_df.drop_duplicates('stems_str')


# In[ ]:


# number of unique songs
print('Number of unique lyrics: {}'.format(sum(artist_text_df['duplicates'] == 0)))
# number of duplicate songs
print('Number of duplicate lyrics: {}'.format(sum(artist_text_df['duplicates'] > 0) +                                               sum(artist_text_df['duplicates'])))
# number of duplicates from same artist
print('Number of duplicate lyrics from same artist: {}'.format(sum(artist_text_df['duplicates'] + 1 -                                                                    artist_text_df['n_artists'])))
# number of duplicates from different artists
print('Number of duplicate lyrics from different artists: {}'.format(sum(artist_text_df['n_artists']                                                                         .loc[artist_text_df['duplicates'] > 0])))


# There are 633 duplicate song texts among them 579 are from different artists. This means that even if the matching of lyrics to artists is perfect it can't reach 100% accuracy. There are 54 duplicate lyrics from the same artist. These lyrics may have different instrumental versions.
# 
# I only considered songs as duplicates when the song texts were completely identical. Possibly there are even more cover songs in the dataset with only very minor differences to the original song. For these songs it will also be very difficult to match the correct artist. 

# <a id="Feature_engineering"></a>
# ## Feature engineering

# In this part I create additional features which can be used by ML algorithm to improve the matching accuracy. 
# 
# To show how these features vary from artist to artist I randomly select a set of 10 artists.

# In[ ]:


# randomly select artists
n_artist = 10
random.seed(0)

artist_select = random.choices(song_df['artist'].unique(), k=n_artist)

song_filter_df = song_df.loc[song_df['artist'].isin(artist_select)]
print('Total number of songs: {}'.format(len(song_filter_df)))
song_filter_df.groupby('artist')[['song']].count().reset_index().rename(columns={'song':'songs'})


# <a id="Number_of_words"></a>
# ### Number of words

# First, I have a look at word counts - are there artists with typically long or short songs?

# In[ ]:


fig = px.box(song_filter_df, x='artist', y='n_stems', title='Word count per song by artist')
fig.show()


# There are some artist who use way more words than others. For example Justin Timberlake as a median of 480 words per song while Harry Belafonte ony has a medain of 168 words. However, there are also many artists with very similar distributions like Kenny Loggins, Louis Jordan, Michael Buble, Supertramp, Tom T. Hall and Vengaboys.

# <a id="Repeated_words"></a>
# ### Repeated words

# Next I check the ratio of unique words over all words. This ratio is 1 if all words of a song are different, the more repeated words appearing in a song the lower this ratio becomes.

# In[ ]:


# number of unique stems
song_df['n_unique_stems'] = song_df['stems'].map(lambda lst: len(set(lst)))
# ratio of unique stems
song_df['unique_stems_ratio'] = song_df['n_unique_stems'] / song_df['n_stems']

# attach column to selected artists
song_filter_df = song_filter_df.join(song_df['unique_stems_ratio'])


# In[ ]:


fig = px.box(song_filter_df, x='artist', y='unique_stems_ratio', title='Ratio of unique words to all words')
fig.show()


# This ratio typically varies from 0.2 to 0.7 among songs of the same artist. Nevertheless, there are some artists who repeat their words more often than others. In this selection these artist are Justin Timberlake and Olly Murs.

# <a id="Words_per_line"></a>
# ### Words per line

# Now I have a look at how many words are there per line. This doesn't say something directly about the content of the songs, however artists may have different styles of structuring their songs. 

# In[ ]:


# calculate number of words per line
song_df['stems_per_line'] = song_df['n_stems'] / song_df['lines'].astype(float)

song_filter_df = song_filter_df.join(song_df[['stems_per_line']])


# In[ ]:


fig = px.box(song_filter_df, x='artist', y='stems_per_line', title='Words per line')
fig.show()


# Tom T. Hall is having many more words per line than all other artists and Vengaboys have clearly fewer words than others. The distributions of all other artist look again very similar.

# <a id="Word_frequency_TFIDF"></a>
# ### Word frequency (TFIDF)

# In this section I use TFIDF to analyse word frequencies in more detail. TFIDF means term frequency - inverse document frequency. Term frequency means how often a word appears in a specific text (in this case song lyrics). Inverse document frequency is the inverse frequency of the same word in the whole document (in this case all song lyrics). 
# 
# The idea of TFIDF is to find out if a specific word appears unusually often in the text. If this is the case, this word gets a high value, if the word appears more often in other texts, its value will be low. 
# 
# This method is very common to cluster texts with similar topics.

# In[ ]:


# initialise count vectorizer
cv = CountVectorizer()

# generate word counts
stem_count_vector = cv.fit_transform(song_df['stems_str'])

# compute idf
tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf_transformer.fit(stem_count_vector)


# In[ ]:


# print idf values
tfidf_df = pd.DataFrame({'stem': cv.get_feature_names(), 'weight': tfidf_transformer.idf_})
 
# get lowest weights
tfidf_df.sort_values('weight').head()


# Above are the words with the lowest values. These words apear many times in almost each song. Thus, these words are not usefull to characterise a song. 

# In[ ]:


# get highest weights
tfidf_df.sort_values('weight', ascending=False).head()


# On the other hand, these are words with the highest values. 
# 
# Next I calculate an TFIDF vector for each song. Every element of this vector represents one word. Its value is calculated by multiplying its frequency with the corresponding weight which was calculated above. This value is normalised by the total number of words of the song. Words which don't appear in the text get a value of 0.
# 
# The TFIDF score is the sum of the TFIDF vector elements. The higher the score the more unusal the words of this song are compared to other songs.

# In[ ]:


# assign tf idf scores to each song
tf_idf_vector = tfidf_transformer.transform(stem_count_vector)

# attach count vectors to dataframe
tf_idf_vector_lst = [-1] * len(song_df)
for i in range(len(song_df)):
    tf_idf_vector_lst[i] = tf_idf_vector[i]
song_df['tf_idf_vector'] = tf_idf_vector_lst    

song_df['tf_idf_score'] = song_df['tf_idf_vector'].map(lambda vec: np.sum(vec.todense()))

# join valus to selected artists
song_filter_df = song_filter_df.join(song_df[['tf_idf_vector', 'tf_idf_score']])


# In[ ]:


fig = px.box(song_filter_df, x='artist', y='tf_idf_score', title='TFIDF scores of songs per artist')
fig.show()


# The distributions are quite different. Vengaboys have the lowest median and a high variance, while for example Supertramp have a high median and a low variance. The artist using the most unusual words is Genesis.
# 
# Another way to quantify differences in word selections is to calculate the similarity of TFIDF vectors. This can be done by calculating the angle between two vectors. For example if `v1` is (1, 0) and `v2` is (0, 1) these vectors point in orthogonal directions and have an angle of 90 degree. The cosine of these vectors would be 0 (this is the lowest possible similarity we can get as all TFIDF vectors cannot have negative values). If we change `v1` to (1, 1) the angle between the vectors would be smaller wich increases the cosine value and therewith their similarity. When two vectors are parallel their angle is 0 which means that the cosine would be 1 which is the maximum similarity. 
# 
# I will apply this metric to TFIDF vectors to compare their similarity. The only difference is that these vectors have several thousand dimensions (one for each word). 
# 
# First I will calculate an TFIDF vector for each artist which is calculated by taking the average of all TFIDF vectors of the artist's songs.

# In[ ]:


# caclculate mean vector
def get_mean_vector(vec_lst):
    return csr_matrix(vstack(vec_lst).mean(axis=0))


# In[ ]:


# calculate mean vector over all songs of same artist
artist_df = song_df.groupby('artist').agg({'tf_idf_vector': get_mean_vector, 'song': len}).reset_index()                   .rename(columns={'song': 'songs'})

# get selected artists
artist_filter_df = artist_df.loc[artist_df['artist'].isin(song_filter_df['artist'])]


# In[ ]:


similarity_matrix = cosine_similarity(vstack(artist_filter_df['tf_idf_vector']), 
                                      vstack(artist_filter_df['tf_idf_vector']))
artist_names = artist_filter_df['artist'].tolist()
fig = go.Figure(data=go.Heatmap(z=np.flipud(similarity_matrix), x=artist_names, y=list(reversed(artist_names)), 
                                colorscale='balance', zmin=0.5, zmax=1.1))
fig.show()


# This matrix visualises the similarity between TFIDF artist vectors. Again, a value of 1 (red) means that the vectors are identical which only appears when comparing vectors of the same artist. The lowest similarities are 0.4 between Louis Jordan and Vengaboys.
# 
# The matrix shows that Vengaboys seem to use very different words than any other artists. We can also see that some artists use very similar words like Kenny Loggins and Genesis. 
# 
# Next, I want to analyse how different TFIDF vectors of songs from the same artist are. Therefore, I calculate the similarity of the artist's TFIDF vector with the TFIDF vector of each song. The problem hereby is that the artist vector was averaged over all song vectors including the one I want to compare against. To avoid any bias from that I calculate the artist vector over all song vectors except the one I'm comparing it against. 
# 
# For example if an artist has three songs: A, B, C. In order to compare how similar song A is with all songs of the artist I calculate the artist vector only from song B and C. To compare song B, the artist vector only consists of song A and C, and so on.

# In[ ]:


artist_song_filter_df = pd.merge(artist_filter_df[['artist', 'tf_idf_vector', 'songs']].assign(key = 0), 
                                 song_filter_df[['artist', 'tf_idf_vector', 'song']].assign(key = 0), on='key', 
                                 suffixes=['_artist', '_song']).drop('key', axis=1).reset_index(drop=True)
artist_song_filter_df['same_artist'] = artist_song_filter_df['artist_artist'] == artist_song_filter_df['artist_song']


# In[ ]:


# calculate similarity of artist tf idf vector and song vector
def tf_idf_vector_similarity(artist_vector, song_vector, songs, same_artist):
    # check if song is from same artist
    if same_artist:
        # deduct song vector from artist vector
        artist_vector = (songs * artist_vector - song_vector) / (songs - 1)
    # calculate similarity
    return cosine_similarity(artist_vector, song_vector)[0][0]


# In[ ]:


artist_song_filter_df['vector_similarity'] =     artist_song_filter_df.apply(lambda row: tf_idf_vector_similarity(row['tf_idf_vector_artist'], 
                                                                     row['tf_idf_vector_song'], 
                                                                     row['songs'], row['same_artist']), axis=1)


# In[ ]:


df = artist_song_filter_df

fig = go.Figure()

fig.add_trace(go.Violin(x=df['artist_artist'][df['same_artist']],
                        y=df['vector_similarity'][df['same_artist']],
                        legendgroup='Same Artist', scalegroup='Same Artist', name='Same Artist',
                        side='negative')
             )
fig.add_trace(go.Violin(x=df['artist_artist'][~df['same_artist']],
                        y=df['vector_similarity'][~df['same_artist']],
                        legendgroup='Different Artists', scalegroup='Different Artists', name='Different Artists',
                        side='positive')
             )

fig.update_traces(meanline_visible=True)
fig.update_layout(violingap=0, violinmode='overlay')
fig.update_layout(title='Similarity of Songs')
fig.update_xaxes(range=[-0.5, 9.5])
fig.update_yaxes(range=[-0.1, 0.8], title='Similarity')
fig.show()


# The distributions show the similarity of song vectors from the same artist with the artist's vector (blue) and the similarity of song vectors from other artists with the artist vector (red). 
# 
# It can be seen that Genesis, Justin Timberlake, Olly Murs, and Supertramp have a much higher similarity with their own songs than with songs from other artists. However other artists like Harry Belafonte, Louis Jordan, and Vengaboys have even a higher similarity with other songs than with their own songs. This is probably a coincidence and its more fair to say that for these artists vector similarity is not a good measure to distinguish their songs from others.

# <a id="Sentiment_analysis"></a>
# ### Sentiment analysis

# The last features I engineer are polarity and subjectivity of songs. The `TextBlob` library has a function to get these values with respect to a given text.

# In[ ]:


polarity_lst = [-1] * len(song_df)
subjectivity_lst = [-1] * len(song_df)
for i, text in enumerate(song_df['text']):
    sentiment = TextBlob(text)
    polarity_lst[i] = sentiment.polarity
    subjectivity_lst[i] = sentiment.subjectivity
    
song_df['polarity'] = polarity_lst
song_df['subjectivity'] = subjectivity_lst

song_filter_df = song_filter_df.join(song_df[['polarity', 'subjectivity']])


# In[ ]:


fig = px.scatter(song_filter_df, x='polarity', y='subjectivity', color='artist', hover_data=['song'], 
                 title='Polarity and Subjectivity of Songs')
fig.show()


# Every point of this plot represents one song. The x value is the polarity and the y value the subjectivity. Although there is a high variance in both features, I cannot find any area with predominantly songs of only one artist. 

# In[ ]:


fig = px.box(song_filter_df, x='artist', y='polarity', title='Polarity by artist')
fig.show()


# The diagram above shows the polarity distribution of songs from the same artist. All distribution look very similar. Thus, neither polarity nor objectivity seems to be a good feature to distinguish songs from different artists.

# <a id="Prediction"></a>
# ## Prediction

# In this part I create a model match song lyrics to their artist. Therefore, I use all song features which were presented above. The artist features are calculated by averaging the features of all songs from this artist. 
# 
# First, I match all songs with all aritsts. Then I use logistic regression to estimate the probability that a song belongs to an artist for each song artist pair. After that I choose for each song the artist with the highest probability.

# I use following variables to create the datasets for training and validating the model:
# - `n_set`: number of sets for training and validation 
# - `n_artist`: number of artists per set
# - `n_song_min`: minimum number of songs an artist must have to be selected
# - `n_song_artist_max`: maximum number of song - artist pairs per artist set
# 
# The artsits are randomly assigned to sets. It is possible that the same artist is assigned to several sets, but it is not possible that there are two sets with identical artists.

# In[ ]:


# parameter
# number of sets
n_set = {'train': 20, 'val': 20}
# number of artists per set
n_artist = 3
# minimum number of songs of one artist
n_song_min = 5
# maximum number of song - artist pairs per artist set
n_song_artist_max = 100


# In[ ]:


def select_artist_song_create_feature(song_df, n_set, n_artist, n_song_min, n_song_artist_max):
    song_count_df = song_df.groupby('artist')[['artist']].count().rename(columns={'artist': 'count'})
    artist_lst = list(song_count_df.loc[song_count_df['count'] >= n_song_min].index.values)

    n_set_total = sum(n_set.values())

    artist_set = []
    while len(artist_set) < n_set_total:
        new_artist = tuple(np.random.choice(artist_lst, size=n_artist, replace=False))
        if new_artist not in artist_set:
            artist_set.append(new_artist)

    # split artist sets
    artist_select = {}
    for field, n in n_set.items():
        i_select = np.random.choice(range(len(artist_set)), size=n, replace=False)
        artist_list = list(artist_set)
        artist_select[field] = [artist_list[i] for i in i_select]
        artist_set = [s for s in artist_set if s not in artist_select[field]]

    # create dataframe with all features
    feature_dict = {}
    # dictionary to map artist set id to list of artists
    set_id_to_artist_tp = {}

    i = 0
    for field, artist_set in artist_select.items():
        df_lst = []
        for artist_tp in artist_set:
            i += 1
            df = song_df.loc[song_df['artist'].isin(artist_tp), 
                             ['artist', 'song', 'n_stems', 'unique_stems_ratio', 'stems_per_line', 'tf_idf_vector', 
                              'tf_idf_score', 'polarity']]
            # check if number of songs is too high
            if len(df) * n_artist > n_song_artist_max:
                df = df.sample(int(n_song_artist_max / n_artist), random_state=0)
                
            df['artist_set_id'] = i
            set_id_to_artist_tp[i] = artist_tp
            df_lst.append(df)
        feature_dict[field] = pd.concat(df_lst)  
        print('Number of songs in {}: {}'.format(field, len(feature_dict[field])))

    # get all selected artists
    artist_select_set = set.union(*[set(sum(tp_lst, ())) for tp_lst in artist_select.values()])

    # create artist dataframe from training data
    df_lst = []
    for artist, df in song_df.loc[song_df['artist'].isin(artist_select_set)].groupby('artist'):
        dic = {'artist': artist}
        # calculate averages and standard diviations
        for field in ['n_stems', 'unique_stems_ratio', 'stems_per_line', 'tf_idf_score', 'polarity']:
            dic[field + '_mean'] = df[field].mean()
            dic[field + '_std'] = df[field].std()

        # number of songs
        dic['songs'] = len(df)

        # calculate average tf idf vector
        dic['tf_idf_vector_mean'] = get_mean_vector(df['tf_idf_vector'])

        df_lst.append(pd.DataFrame(dic, index=[0]))
    artist_feature_df = pd.concat(df_lst)

    def get_features(df):
        # get artist set id
        artist_set_id = df['artist_set_id'].iloc[0]
        
        # get all artists
        artist_feature_select_df = artist_feature_df.loc[artist_feature_df['artist']                                                         .isin(set_id_to_artist_tp[artist_set_id])]

        # merge dataframes
        artist_song_feature_df = pd.merge(artist_feature_select_df.assign(key=0), df.assign(key=0), on='key', 
                                          suffixes=['_artist', '_song']).drop('key', axis=1)    
        artist_song_feature_df['same_artist'] =             artist_song_feature_df['artist_artist'] == artist_song_feature_df['artist_song']

        # calculate features
        for feature in ['n_stems', 'unique_stems_ratio', 'stems_per_line', 'tf_idf_score', 'polarity']:
            artist_song_feature_df[feature + '_diff'] =                 artist_song_feature_df[feature] - artist_song_feature_df[feature + '_mean']
            artist_song_feature_df[feature + '_diff_std'] =                 artist_song_feature_df[feature + '_diff'] / artist_song_feature_df[feature + '_std']

        # calculate vector similarity between artist and song
        artist_song_feature_df['vector_similarity'] =             artist_song_feature_df.apply(lambda row: tf_idf_vector_similarity(row['tf_idf_vector_mean'], 
                                                      row['tf_idf_vector'], row['songs'], row['same_artist']), 
                                         axis=1)    
        return artist_song_feature_df

    artist_song_feature = {}
    for field in feature_dict:
        artist_song_feature[field] = feature_dict[field].groupby('artist_set_id').apply(get_features)                                                        .reset_index(drop=True)
        
    return artist_song_feature


# In[ ]:


np.random.seed(0)
artist_song_feature = select_artist_song_create_feature(song_df, n_set, n_artist, n_song_min, n_song_artist_max)


# The above function creates a dictionary with the fields specified in `n_set` with the number of defined artist sets. An artist set is a set of songs from `n_artist` number of randomly selected artists. 
# 
# The values of this dictionary are dataframes with all artist sets, e.g. if `n_set['train'] = 20` it contains 20 artist sets. Each row of the data set contains a pair of artist - song matches. Thereby, the song is from one of the randomly selected artists of this set. In this example the first artist set contains the artists: Little Mix, Our Lady Peace, and Underoath.

# In[ ]:


artist_song_feature['train'].iloc[0]


# Above are all columns with values of the first row of the dataframe. Every artist set has an id which is contained in `artist_set_id`. The column `artist_artist` contains the name of the artist from whom all artist related features were taken. `aritst_song` is the name of the artist whoes song was matched to the artist. In this case the artist features are from "Little Mix" and the song features are from the song "Secret Love" from "Little Mix". Thus, the prediction algorithm is expected to return a high probability that the artist and song match (the target variable is `same_artist`). 
# 
# Following features are added to each artist and song: `n_stems`, `unique_stems_ratio`, `stems_per_line`, `tf_idf_score`, and `polarity`. The artist features contain the mean (`_mean`) and standard deviation (`_std`). The difference of the song and artist features (song feature - artist feature) have the suffix `_diff`. Additionally these features are divided by the standard deviation to get a normalised measure for the difference (`_diff_std`). The dataframe also contains the TFIDF vector of all artist songs and the matched song (`tf_idf_vector_mean` and `tf_idf_vector`), the similarity of the vectors contains the feature `vector_similarity`. 

# In[ ]:


feature = ['n_stems_diff', 'n_stems_diff_std',
       'unique_stems_ratio_diff', 'unique_stems_ratio_diff_std',
       'stems_per_line_diff', 'stems_per_line_diff_std', 'tf_idf_score_diff',
       'tf_idf_score_diff_std', 'polarity_diff', 'polarity_diff_std',
       'vector_similarity']
df_lst = []
for f in feature:
    df = artist_song_feature['train'][['same_artist']]
    df['feature'] = f
    df['value'] = artist_song_feature['train'][f]
    df_lst.append(df)
feature_df = pd.concat(df_lst)
feature_df.head()


# In[ ]:


def violine_feature_plot(feature_df, feature_select):

    fig = go.Figure()
    df = feature_df.loc[feature_df['feature'].isin(feature_select)]

    fig.add_trace(go.Violin(x=df['feature'][df['same_artist']],
                            y=df['value'][df['same_artist']],
                            legendgroup='Same Artist', scalegroup='Same Artist', name='Same Artist',
                            side='negative')
                 )
    fig.add_trace(go.Violin(x=df['feature'][~df['same_artist']],
                            y=df['value'][~df['same_artist']],
                            legendgroup='Different Artists', scalegroup='Different Artists', name='Different Artists',
                            side='positive')
                 )

    fig.update_traces(meanline_visible=True)
    fig.update_layout(violingap=0, violinmode='overlay')
    fig.update_layout(title='Feature Comparison')
    fig.update_xaxes(title='Feature')
    return fig


# In[ ]:


fig = violine_feature_plot(feature_df, ['n_stems_diff_std', 'unique_stems_ratio_diff_std', 'stems_per_line_diff_std', 
                                        'tf_idf_score_diff_std', 'polarity_diff_std'])
fig.update_xaxes(range=[-0.5, 4.5])
fig.show()


# The violine plot above shows the distribution of the normalised difference features `n_stems`, `unique_stems_ratio`, `stems_per_line`, `tf_idf_score`, and `polarity`. There is one distribution created which only contains artist - song pairs of the same artist (blue) and one distribution for the case of different artists (red). 
# 
# All distributions look very similar, it is very difficult to find only one feature which indicates a difference between the same and different artists. 

# In[ ]:


fig = violine_feature_plot(feature_df, ['vector_similarity'])
fig.update_xaxes(range=[-1, 1])
fig.show()


# This plot shows the distribution for the similarity of TFIDF vectors. Here we can see a difference in the distributions for songs from the same artist or different artists. So this seems to be the best feature. However, the distributions overlap a lot, therefore it will be difficult in general to distinguish songs from the same or different artist.

# In[ ]:


def prepare_data(df, feature_org, feature_abs):
    for f in feature_abs:
        df[f] = df[f].abs()
    X = df[feature_org + feature_abs].values
    y = df['same_artist'].values
    
    return X, y

def select_songs_train_pipeline(song_df, n_set, n_artist, n_song_min, n_song_artist_max, feature_org, feature_abs, 
                                pipeline):
    artist_song_feature = select_artist_song_create_feature(song_df, n_set, n_artist, n_song_min, n_song_artist_max)

    # prepare data
    X, y = prepare_data(artist_song_feature['train'], feature_org, feature_abs)

    pipeline = pipeline.fit(X, y)
    
    return artist_song_feature, pipeline


# Above I created one function to convert the extracted feature values to a matrix and get a target vector for prediction. The other function combines the creation of the feature dataframe, convertion of the features to a matrix and training a machine learning pipeline.

# In[ ]:


# prepare data create and train pipeline
n_artist = 3
n_song_min = 5
n_set = {'train': 100}
n_song_artist_max = 100

feature_org = ['n_stems', 'unique_stems_ratio', 'stems_per_line', 'tf_idf_score', 'polarity', 'vector_similarity']
feature_abs = ['n_stems_diff', 'n_stems_diff_std', 'unique_stems_ratio_diff', 'unique_stems_ratio_diff_std', 
               'stems_per_line_diff', 'stems_per_line_diff_std', 'tf_idf_score_diff', 'tf_idf_score_diff_std', 
               'polarity_diff', 'polarity_diff_std']

pipeline = Pipeline([('scale', StandardScaler()), 
                     ('clf', LogisticRegression(solver='lbfgs', max_iter=3000, 
                                                class_weight={False: 1/n_artist, True:(n_artist - 1)/n_artist}))])

np.random.seed(1)
artist_song_feature, pipeline = select_songs_train_pipeline(song_df, n_set, n_artist, n_song_min, n_song_artist_max, 
                                                            feature_org, feature_abs, pipeline)


# I selected all features which were introduced in the feature engineering part, created a pipeline which normalises all features and uses logistic regression to estimate the probabilities. In the definition of the classifier I use class weights: positive samples (song is from matched artist) are weithted with `(n_artist - 1)/n_artist` (`n_artist` is the number of artists per set) while negative samples are weithed with `1/n_artist`. Thus, the more artists are in the set the higher the weight for positive samples. The higher the weight the stronger the impact of a wrong prediction on the loss function. For example in case of 5 artists predicting a true sample as false has a five times higher impact than predicting a false sample as true. This is done to prevent the classifier from predicting every sample as false. Therefore, a classifier which would predict every sample as false would get an accuracy of 80% while the weighted accuracy would be only 50%. 
# 
# I also tried other classifiers like random forest, support vector machine, or XG boost but always got very similar results. Therefore, I decided to use the simplest classifier. 

# In[ ]:


feature_importance_df = pd.DataFrame({'feature': feature_org+feature_abs, 'coefficient':pipeline['clf'].coef_[0]})

px.bar(feature_importance_df.sort_values('coefficient'), x='feature', y='coefficient')


# The plot shows the coefficient values of the logistic regression model. The higher the absolute value of the coefficient, the more important the feature. Thus, the features at the edge of the plot are the most important ones. 
# 
# The more positive a coeffiecient the stronger the corresponding feature (all features were normalised and have positive values) causes a positive prediction. Thus, if the features on the left side of the plot have high values a negative prediction (different artists) is more likely, while the higher the features on the right the more likely a positive prediction (same artists) becomes.
# 
# The most negative coefficient is for `tf_idf_score` this means the more unusual the words of the song the more likely the song is from a different artist. Although this affects the probability, this feature only depends on the song and therefore does not affect the selection of artist - song pair.
# The next feature is `stems_per_line_diff_std`, this feature describes the difference in the number of stems (words) per line between the artist and song. The higher the difference the more unlikely that the artist matches to the song. 
# 
# On the other end of the scale is `vector_similarity`. This feature describes how similar the TF IDF vectors of the artist and the song are. The more similar the more likely it is that the artist and song match.

# <a id="Validation"></a>
# ## Validation

# In[ ]:


def predict_artist(df, feature_org, feature_abs, pipeline, top_n):
    # prepare data
    X, y = prepare_data(df, feature_org, feature_abs)
    
    # get probability
    proba = pipeline.predict_proba(X)
    # attach to dataframe
    df['probability'] = proba[:, 1]
    df['correct_prediction'] = df['artist_artist'] == df['artist_song']
    
    # get artist song pairs with highest probability
    predict_select = df.sort_values('probability', ascending=False).groupby(['artist_set_id']).head(top_n)                       .groupby(['artist_set_id'])['correct_prediction'].max()
    
    # get accuracy
    print('Accuracy: {}'.format(predict_select.mean()))
    
    return predict_select


# The function above makes the matching prediction and validates the prediction accuracy. The variable `top_n` specifies how many top predictions are considered as correct. For example if there are four artists (A, B, C, D) matched to one song which belongs to artist C, the model orders the artist with repspect to the probability that they match to the respective song. The result could be B, C, A, D. 
# 
# If `top_n` is set to one, the prediction is only considered as correct if the artist with the highest probability (in this case B) matches to the song. As this is not the case the prediction would be considered as wrong. If `top_n` is set to 2 or higher the prediction would be correct as C is the artist with the second highest probability. 
# Hence, the higher `top_n` the more likely a prediction is considered as correct. 

# In[ ]:


artist_predict_df = predict_artist(artist_song_feature['train'], feature_org, feature_abs, pipeline, top_n=1)


# In[ ]:


artist_predict_df = predict_artist(artist_song_feature['train'], feature_org, feature_abs, pipeline, top_n=2)


# If we only accept the artist with the highest probability as correct (`top_n = 1`) the training accuracy is 79%. Increasing `top_n` to 2 raises the accuracy to 97%.

# In[ ]:


n_artist_lst = [2, 4, 8, 16, 32, 64, 128]
top_n_lst = [1, 2, 4, 8, 16, 32, 64]
n_song_artist_max = 128
np.random.seed(2)

n_set = {'train': 100, 'val': 100}

feature_org = ['n_stems', 'unique_stems_ratio', 'stems_per_line', 'tf_idf_score', 'polarity', 'vector_similarity']
feature_abs = ['n_stems_diff', 'n_stems_diff_std', 'unique_stems_ratio_diff', 'unique_stems_ratio_diff_std', 
               'stems_per_line_diff', 'stems_per_line_diff_std', 'tf_idf_score_diff', 'tf_idf_score_diff_std', 
               'polarity_diff', 'polarity_diff_std']

pipeline = Pipeline([('scale', StandardScaler()), 
                     ('clf', LogisticRegression(solver='lbfgs', max_iter=3000, 
                                                class_weight={False: 1/n_artist, True:(n_artist - 1)/n_artist}))])

result_lst = []

for n_artist in n_artist_lst:
    print(datetime.now())
    print('n_artist: {}'.format(n_artist))
    
    artist_song_feature, pipeline = select_songs_train_pipeline(song_df, n_set, n_artist, n_song_min, 
                                                                n_song_artist_max, feature_org, feature_abs, pipeline)
    
    for top_n in [n for n in top_n_lst if n < n_artist]:
        print('top_n: {}'.format(top_n))
        
        predict_select = predict_artist(artist_song_feature['val'], feature_org, feature_abs, pipeline, top_n=top_n)
        
        result_dict = {'n_artist': n_artist, 'top_n': top_n, 'accuracy': predict_select.mean()}
        result_lst.append(result_dict)
        
    print('')
    
result_df = pd.DataFrame(result_lst)


# In[ ]:


fig = px.line(result_df, x='n_artist', y='accuracy', color='top_n', 
              title='Accuracy vs number of artist and number of top selections', 
              labels={'n_artist': 'Number of artists per set', 'top_n': 'Top predictions'})\
        .update_traces(mode='lines+markers')
fig.show()


# To validate the performance of the model I randomly create 100 training and 100 valdiation sets of 2, 4, 8, 16, 32, 64, and 128 artists per set. The accuracy of the model is calculated for values of `top_n` from 1 until 64. 
# 
# The graph above shows how the accrucy decreased with respect to the number of artists for different values of `top_n`. If a match is only considered as correct when the artist of a song is the artist with the highest probability (`top_n = 1`) then the accuracy decreases from 96% for two artists per set to 12% for sets of 128 artists. Although 12% is a low value, it must be kept in mind that the probability of selection the correct artist purly by guessing would be 1/128 which is about 0.8%. 
# 
# When a match is considered as correct if the selected artist is within the upper half of all artists (e.g. for a set of 32 artist `top_n` is set to 16). Then the accuracy ranges from 90% to 97%.

# <a id="Potential_improvements"></a>
# ## Potential improvements

# I expect that the most likely way to improve the model's accuracy would be to add more features to the songs/artits. Probably the most important feature is the vector similarity which is based on the TFIDF vectors of each song lyric. These vectors have more than 50,000 dimensions (one for each unique stems). Therefore, it is not practical to create one feature for each vector element. However, a way to extract more compact information could be to do PCA (princile component analysis) and select the eigenvalues of the most important components. 
# 
# Another possiblity to create more features would be to apply `doc2vec` which is a neural networks based approach to convert a text to a vector.
