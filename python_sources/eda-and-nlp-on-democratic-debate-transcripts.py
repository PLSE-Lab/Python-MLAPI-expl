#!/usr/bin/env python
# coding: utf-8

# **Let's conduct some exploratory data analysis and sentiment analysis on the debate transcripts and see what we find.**
# 
# **Credit**
# 
# Thanks to Branden Ciranni for posting the debate transcripts
# https://www.kaggle.com/brandenciranni/democratic-debate-transcripts-2020
# 
# Thanks to Oumainma Hourrance for posting the sentiment analysis training data
# https://www.kaggle.com/oumaimahourrane/imdb-reviews
# 
# **Workflow**
# 
# 1. Exploratory data analysis
# 2. Basic NLP on speech text
# 3. Review word usage
# 4. Train a model to detect sentiment
# 5. Conduct sentiment analysis on speech text

# In[ ]:


import numpy as np
import pandas as pd 

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Input, Embedding, Dense, Dropout, BatchNormalization, Activation, Bidirectional, LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

import re
from operator import itemgetter
import collections

import nltk
from nltk.stem import PorterStemmer
from nltk.util import ngrams
from nltk import pos_tag
from nltk import RegexpParser
nltk.download('stopwords')

import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud

from collections import defaultdict

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


df = pd.read_csv("/kaggle/input/democratic-debate-transcripts-2020/debate_transcripts_v3_2020-02-26.csv", encoding="cp1252")


# In[ ]:


df.head()


# In[ ]:


df.shape


# In[ ]:


# Check for null values
df.isna().sum()


# In[ ]:


# Drop rows that do not contain a speaking time
df.dropna(inplace=True)


# In[ ]:


# Review all sections
df.debate_section.unique()


# In[ ]:


# Review all speakers
df.speaker.unique()


# In[ ]:


df.speaker.value_counts()


# In[ ]:


df_total_speaking_time = df.groupby(df.speaker)["speaking_time_seconds"].sum().sort_values()
# Review mean and median total speaking time
df_total_speaking_time.mean(), df_total_speaking_time.median()


# In[ ]:


# Let's drop speakers who had limited speaking time and view the remaining speakers
df = df[df.groupby(df.speaker)["speaking_time_seconds"].transform("sum") > 1100]
df.speaker.unique()


# In[ ]:


plt.figure(figsize=(20,7))
plt.xticks(fontsize=20)
plt.yticks(fontsize=15)
plt.ylabel('Total Speaking Time', fontsize=20)
plt.xlabel('Speaker', fontsize=20)
df.groupby(df.speaker)["speaking_time_seconds"].sum().sort_values().plot.bar()


# In[ ]:


# add a column for the speech with stop words and punctuation removed
stop_words = set(nltk.corpus.stopwords.words('english'))
for word in ["its", "would", "us", "then", "so", "it", "thats", "going", "also"]:
    stop_words.add(word)
df["speech_cleaned"] = df["speech"].apply(lambda x: " ".join([re.sub(r'[^\w\d]','', item.lower()) for item in x.split() if re.sub(r'[^\w\d]','', item.lower()) not in stop_words]))


# **Word Usage**

# In[ ]:


# Let's look the most words used. 'People' has been by far the most used word.
t = Tokenizer()
t.fit_on_texts(df.speech_cleaned)
top_20_words = sorted(t.word_counts.items(), key=itemgetter(1), reverse=True)[:20]
top_20_words


# In[ ]:


# Create a word cloud with the most used words
wordcloud = WordCloud()
wordcloud.generate_from_frequencies(dict(top_20_words))
plt.figure(figsize=(12,10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


# In[ ]:


# Create a tokenizer for each candidate in order to create a bag of words for each.
tokenizers = defaultdict(Tokenizer)
for name in df.speaker.unique():
    tokenizers[name].fit_on_texts(df.speech_cleaned[df.speaker == name])


# In[ ]:


for candidate in df.speaker.unique():
    print(candidate , "\n", sorted(tokenizers[candidate].word_counts.items(), key=itemgetter(1), reverse = True)[:10], "\n")


# One thing that jumped out to me was how often Biden uses the word "fact", so I compared it across the other cadidates below. Out of curiousity, I also compared use of the word 'Trump'.

# In[ ]:


fact_dict = dict()
for candidate in df.speaker.unique():
    if "fact" in tokenizers[candidate].word_index:
        fact_dict[candidate] = tokenizers[candidate].word_counts["fact"]
    else:
        fact_dict[candidate] = 0
sorted(fact_dict.items(), key=itemgetter(1), reverse=True)


# In[ ]:


# Healthcare appears to be a common word as well, let's see which candidates speak use the word 'healthcare' most often
healthcare_dict = dict()
for candidate in df.speaker.unique():
    if "healthcare" in tokenizers[candidate].word_index:
        healthcare_dict[candidate] = tokenizers[candidate].word_counts["healthcare"]
    else:
        healthcare_dict[candidate] = 0
sorted(healthcare_dict.items(), key=itemgetter(1), reverse=True)


# Unsurprisingly, Bernie Sanders and Elizabeth Warren use the word 'healthcare' the most. It is worth noting, that Michael Bloomberg did not use the word once.

# **N-grams**

# In[ ]:


# Tokenize all text in order
text = ""
tokenized = list()
for speech in df.speech_cleaned:
    text += " " + speech
tokenized = text.split()


# In[ ]:


# Most common bi-grams
n_grams = collections.Counter(ngrams(tokenized, 2))
n_grams.most_common(10)


# In[ ]:


# Most common tri-grams
n_grams = collections.Counter(ngrams(tokenized, 3))
n_grams.most_common(10)


# In[ ]:


# Most common quad-grams
n_grams = collections.Counter(ngrams(tokenized, 4))
n_grams.most_common(10)


# In[ ]:


# Most common quint-grams
n_grams = collections.Counter(ngrams(tokenized, 5))
n_grams.most_common(10)


# Top Joe Biden N-grams

# In[ ]:


# Tokenize all Joe Biden text in order
text = ""
tokenized = list()
for speech in df.speech_cleaned[df.speaker=="Joe Biden"]:
    text += " " + speech
tokenized = text.split()


# In[ ]:


# Most common Biden bi-grams
n_grams = collections.Counter(ngrams(tokenized, 2))
n_grams.most_common(10)


# In[ ]:


# Most common Biden tri-grams
n_grams = collections.Counter(ngrams(tokenized, 3))
n_grams.most_common(10)


# Top Bernie Sanders N-grams

# In[ ]:


# Tokenize all Bernie Sanders text in order
text = ""
tokenized = list()
for speech in df.speech_cleaned[df.speaker=="Bernie Sanders"]:
    text += " " + speech
tokenized = text.split()


# In[ ]:


# Most common Bernie Sanders bi-grams
n_grams = collections.Counter(ngrams(tokenized, 2))
n_grams.most_common(10)


# In[ ]:


# Most common Bernie Sanders tri-grams
n_grams = collections.Counter(ngrams(tokenized, 3))
n_grams.most_common(10)


# **Sentiment Analysis**
# 
# Let's train a model to detect sentiment, then examine how positive or negative each speaker is. For this, we use IMDB movie reviews as training data. A smaller dataset was selected to keep training time down, but it should be good enough to provide an overview.

# In[ ]:


# review the speech lengths
df_speech_length = df["speech"].apply(lambda x: len(x.split()))
df_speech_length.hist(bins=30)
df_speech_length.mean(), df_speech_length.median(), np.percentile(df_speech_length, 80)


# In[ ]:


# set max sequence length
max_len = 150


# In[ ]:


# Load reviews for sentiment analysis
df_reviews = pd.read_csv("../input/imdb-reviews/dataset.csv", encoding="cp1252")


# In[ ]:


df_reviews.head()


# In[ ]:


# Create stemmer
stemmer = PorterStemmer()


# In[ ]:


# Clean up review text
df_reviews["SentimentTextCleaned"] = df_reviews["SentimentText"].apply(lambda x: " ".join([stemmer.stem(re.sub(r'[^\w\d]','', item.lower())) for item in x.split() if re.sub(r'[^\w\d]','', item.lower()) not in stop_words]))                                                                                                      


# In[ ]:


# Take a look at cleaned up reviews
df_reviews["SentimentTextCleaned"][:10]


# In[ ]:


review_tokenize = Tokenizer()
review_tokenize.fit_on_texts(df_reviews["SentimentTextCleaned"])
X_sentiment = pad_sequences(review_tokenize.texts_to_sequences(df_reviews["SentimentTextCleaned"]), maxlen=max_len, padding="post")
Y_sentiment = df_reviews["Sentiment"]


# In[ ]:


# build a model for sentiment analysis
sentiment_model = Sequential([
    Embedding(len(review_tokenize.word_index) + 1, 64),
    Bidirectional(LSTM(32, return_sequences=True)),
    Bidirectional(LSTM(16)),
    Dense(64),
    BatchNormalization(),
    Activation("relu"),
    Dropout(.25),
    Dense(16),
    BatchNormalization(),
    Activation("relu"),
    Dropout(.25),
    Dense(2, activation="softmax")
])


# In[ ]:


sentiment_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["acc"])


# In[ ]:


sentiment_model.fit(X_sentiment, Y_sentiment, validation_split=.1, epochs=4)


# In[ ]:


# Stem cleaned up speech in the debate data
df["speech_cleaned"] = df["speech_cleaned"].apply(lambda x: " ".join([stemmer.stem(item) for item in x.split()]))
# Review stemmed speech
df["speech_cleaned"][:10]


# In[ ]:


# Add a column to show the sentiment of each speech
predictions = []
for speech in df["speech_cleaned"]:
  prediction = sentiment_model.predict(pad_sequences(review_tokenize.texts_to_sequences([speech]), maxlen=max_len, padding="post"))
  predictions.append(prediction[0][1])
df["sentiment"] = predictions


# In[ ]:


# Review sentiment by speaker. The higher the number the more positive their speech is.
plt.figure(figsize=(20,7))
plt.xticks(fontsize=20)
plt.yticks(fontsize=15)
plt.ylabel("Average Sentiment", fontsize=20)
plt.xlabel("Speaker", fontsize=20)
df.groupby(df.speaker)["sentiment"].mean().sort_values().plot.bar()

