#!/usr/bin/env python
# coding: utf-8

# ## Abstract
# 
# ### Can you identify Sarcastic sentences?
# ### Can you distinguish between Fake news and Legitimate news?
# 
# Since news headlines are written by professionals in a formal manner, there are no spelling mistakes and informal usage. This reduces the sparsity and also increases the chance of finding pre-trained embeddings.
# 
# Furthermore, since the sole purpose of TheOnion is to publish sarcastic news, we get high-quality labels with much less noise as compared to Twitter datasets.
# 
# ### Dataset
# 
# Each record consists of three attributes:
# 
# is_sarcastic: 1 if the record is sarcastic otherwise 0
# 
# headline: the headline of the news article
# 
# article_link: link to the original news article. Useful in collecting supplementary data
# 
# ### Deep Learning Framework
# I will be using Tensorflow to implement the Deep Learning Models for this Project.

# ### Import the libraries

# In[ ]:


import json
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from nltk.stem.wordnet import WordNetLemmatizer
import re, string
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ReduceLROnPlateau 


import warnings
warnings.filterwarnings("ignore")


# ### Loading the Dataset

# In[ ]:


df_News = pd.read_json('../input/news-headlines-dataset-for-sarcasm-detection/Sarcasm_Headlines_Dataset_v2.json', lines=True)
df_2 = pd.read_json('../input/news-headlines-dataset-for-sarcasm-detection/Sarcasm_Headlines_Dataset.json', lines=True)
df_News.head()


# ### Joining the two Datasets

# In[ ]:


df_News = pd.concat([df_News, df_2], ignore_index=True)
df_News.shape


# ### Exploratory Data Analysis

# In[ ]:


df_News.info()


# In[ ]:


# looking at some sarcastic news
df_News[df_News.is_sarcastic == 1].head(5)


# In[ ]:


# looking at some legitimate news
df_News[df_News.is_sarcastic == 0].head(5)


# ### Target Column Distribution

# In[ ]:


df_News.is_sarcastic.value_counts()


# The dataset appears to be balanced for Sarcastic and legitimate news

# In[ ]:


sns.countplot(df_News.is_sarcastic)


# In[ ]:


wordcloud = WordCloud(background_color='black',
                    stopwords = STOPWORDS,
                    max_words = 100,
                    random_state = 101, 
                    width=1800, 
                    height=1000)
wordcloud.generate(str(df_News['headline']))
plt.imshow(wordcloud)


# ### News Headline length Distribution

# In[ ]:


df_News['headline_len'] = df_News.headline.apply(lambda x: len(x.split()))


# In[ ]:


sarcastic = df_News[df_News.is_sarcastic == 1]
legit = df_News[df_News.is_sarcastic == 0]


# In[ ]:


plt.figure(figsize=(8,5))
sns.distplot(sarcastic.headline_len, hist= True, label= 'Sarcastic')
sns.distplot(legit.headline_len, hist= True, label= 'legitimate')
plt.legend()
plt.title('News Headline Length Distribution by Class', fontsize = 10)
plt.show()


# ### Data Cleaning & Pre-processing

# I will be doing just some basic Pre-processing, as these are News Headlines which are written by Professionals in a formal way, so to not remove anything which can help with context, I will only remove punctuations and apply lemmatization

# In[ ]:


df_News = df_News.drop(columns=['article_link'])


# In[ ]:


lem = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))
punctuations = string.punctuation


# In[ ]:


def clean_text(news):
    """
    This function receives headlines sentence and returns clean sentence
    """
    news = news.lower()
    news = re.sub("\\n", "", news)
    #news = re.sub("\W+", " ", news)
    
    #Split the sentences into words
    words = list(news.split())
    
    words = [lem.lemmatize(word, "v") for word in words]
    words = [w for w in words if w not in punctuations]
    #words = [w for w in words if w not in stop_words]
    #words = [''.join(x for x in w if x.isalpha()) for w in words]

    clean_sen = " ".join(words)
    
    return clean_sen


# In[ ]:


df_News['news_headline'] = df_News.headline.apply(lambda news: clean_text(news)) 
df_News.head()


# In[ ]:


df_News.groupby(['is_sarcastic']).headline_len.mean()


# In[ ]:


df_News.groupby(['is_sarcastic']).headline_len.max()


# On an Average most of the headlines have same length, in some cases, Sarcastic headlines are longer. 

# ### Stratified Split 
# 
# I am using Stratified split to sample approx equal number of instances for training for both the categories of our Target.

# In[ ]:


headlines = df_News['news_headline']
labels = df_News['is_sarcastic'] 


# In[ ]:


train_sentences, test_sentences, train_labels, test_labels = train_test_split(headlines, labels, test_size=0.2, stratify=labels, random_state=42)


# In[ ]:


train_labels.value_counts()


# ### Converting News Headlines into Sequences of tokens

# In[ ]:


#Defining Hyperparameters to be used

max_words = 30000     # how many unique words to use (i.e num rows in embedding vector)
max_len = 70       # max number of words in a headline to use
oov_token = '00_V'    # for the words which are not in training samples
padding_type = 'post'   # padding type
trunc_type = 'post'    # truncation for headlines longer than max length
embed_size = 100    # how big is each word vector


# In[ ]:


tokenizer = Tokenizer(num_words=max_words, oov_token=oov_token)
tokenizer.fit_on_texts(train_sentences)

word_index = tokenizer.word_index


# In[ ]:


train_sequences = tokenizer.texts_to_sequences(train_sentences)
train_sequences = pad_sequences(train_sequences, maxlen=max_len, padding=padding_type, truncating=trunc_type)

test_sequences = tokenizer.texts_to_sequences(test_sentences)
test_sequences = pad_sequences(test_sequences, maxlen=max_len, padding=padding_type, truncating=trunc_type)


# In[ ]:


train_sequences


# ### Predictive Modeling
# Our aim is to build a binary Classification model which given a sequence of text, can classify it as Sarcastic or not, or Fake news or real news.
# 
# ### A. I will try below different models and see which works best.
# 1. Neural Network with Embedding
# 2. RNN(Recurrent Neural Network)
# 3. LSTM(Long-short term Memory) with GlobalAveragePooling
# 4. LSTM with GlobalMaxPooling
# 5. Stacked LSTM
# 6. Bidirectional LSTM
# 7. GRU(Gated Recurrent Unit)
# 8. Stacked Bidirectional LSTM
# 9. Best Model from above with pre-trained Embeddings

# ### 1. Neural Network with Embedding

# In[ ]:


from tensorflow.keras.callbacks import ReduceLROnPlateau 

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(max_words, embed_size, input_length=max_len),
    tf.keras.layers.GlobalMaxPooling1D(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])


rlrp = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()


# In[ ]:


history = model.fit(train_sequences, train_labels, batch_size=32, epochs=5, 
                    validation_data=(test_sequences, test_labels), 
                    callbacks=[rlrp] ,verbose=1)


# In[ ]:


score = model.evaluate(test_sequences, test_labels)
print('Test Loss: ', score[0])
print('Test Accuracy', score[1])


# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# ### 2. RNN (Recurrent Neural Network)

# In[ ]:


model_rnn = tf.keras.Sequential([
    tf.keras.layers.Embedding(max_words, embed_size, input_length=max_len),
    tf.keras.layers.SpatialDropout1D(0.2),
    tf.keras.layers.SimpleRNN(32, dropout=0.2, recurrent_dropout=0.2, return_sequences=True),
    tf.keras.layers.GlobalMaxPooling1D(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])


rlrp = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2)
model_rnn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model_rnn.summary()


# In[ ]:


history_rnn = model_rnn.fit(train_sequences, train_labels, batch_size=32, epochs=5, 
                    validation_data=(test_sequences, test_labels), 
                    callbacks=[rlrp] ,verbose=1)


# In[ ]:


score = model_rnn.evaluate(test_sequences, test_labels)
print('Test Loss: ', score[0])
print('Test Accuracy', score[1])


# list all data in history
print(history_rnn.history.keys())
# summarize history for accuracy
plt.plot(history_rnn.history['accuracy'])
plt.plot(history_rnn.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history_rnn.history['loss'])
plt.plot(history_rnn.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# ### 3. LSTM (Long Short Term Memory) with GlobalMaxPooling & SpatialDropout

# In[ ]:


from tensorflow.keras.callbacks import ReduceLROnPlateau 

model_lstm = tf.keras.Sequential([
    tf.keras.layers.Embedding(max_words, embed_size, input_length=max_len),
    tf.keras.layers.SpatialDropout1D(0.2),
    tf.keras.layers.LSTM(32, dropout=0.2, recurrent_dropout=0.2, return_sequences=True),
    tf.keras.layers.GlobalMaxPooling1D(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])


rlrp = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2)
model_lstm.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model_lstm.summary()


# In[ ]:


history_lstm = model_lstm.fit(train_sequences, train_labels, batch_size=32, epochs=5, 
                    validation_data=(test_sequences, test_labels), 
                    callbacks=[rlrp] ,verbose=1)


# In[ ]:


score = model_lstm.evaluate(test_sequences, test_labels)
print('Test Loss: ', score[0])
print('Test Accuracy', score[1])


# list all data in history
print(history_lstm.history.keys())
# summarize history for accuracy
plt.plot(history_lstm.history['accuracy'])
plt.plot(history_lstm.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history_lstm.history['loss'])
plt.plot(history_lstm.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# ### 4. LSTM with GlobalAveragePool

# In[ ]:


from tensorflow.keras.callbacks import ReduceLROnPlateau 

model_lstm_avg = tf.keras.Sequential([
    tf.keras.layers.Embedding(max_words, embed_size, input_length=max_len),
    tf.keras.layers.SpatialDropout1D(0.2),
    tf.keras.layers.LSTM(32, dropout=0.2, recurrent_dropout=0.2, return_sequences=True),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])


rlrp = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2)
model_lstm_avg.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model_lstm_avg.summary()


# In[ ]:


history_lstm_avg = model_lstm_avg.fit(train_sequences, train_labels, batch_size=32, epochs=5, 
                    validation_data=(test_sequences, test_labels), 
                    callbacks=[rlrp] ,verbose=1)


# In[ ]:


score = model_lstm_avg.evaluate(test_sequences, test_labels)
print('Test Loss: ', score[0])
print('Test Accuracy', score[1])


# list all data in history
print(history_lstm_avg.history.keys())
# summarize history for accuracy
plt.plot(history_lstm_avg.history['accuracy'])
plt.plot(history_lstm_avg.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history_lstm_avg.history['loss'])
plt.plot(history_lstm_avg.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# ### 5. LSTM with only one FC Dense Layer

# In[ ]:


from tensorflow.keras.callbacks import ReduceLROnPlateau 

model_lstm1 = tf.keras.Sequential([
    tf.keras.layers.Embedding(max_words, embed_size, input_length=max_len),
    tf.keras.layers.SpatialDropout1D(0.2),
    tf.keras.layers.LSTM(32, dropout=0.2, recurrent_dropout=0.2, return_sequences=True),
    tf.keras.layers.GlobalMaxPooling1D(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])


rlrp = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2)
model_lstm1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model_lstm1.summary()


# In[ ]:


history_lstm1 = model_lstm1.fit(train_sequences, train_labels, batch_size=32, epochs=5, 
                    validation_data=(test_sequences, test_labels), 
                    callbacks=[rlrp] ,verbose=1)


# In[ ]:


score = model_lstm1.evaluate(test_sequences, test_labels)
print('Test Loss: ', score[0])
print('Test Accuracy', score[1])


# list all data in history
print(history_lstm1.history.keys())
# summarize history for accuracy
plt.plot(history_lstm1.history['accuracy'])
plt.plot(history_lstm1.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history_lstm1.history['loss'])
plt.plot(history_lstm1.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# ### Stacked LSTM

# In[ ]:


from tensorflow.keras.callbacks import ReduceLROnPlateau 

model_st_lstm = tf.keras.Sequential([
    tf.keras.layers.Embedding(max_words, embed_size, input_length=max_len),
    tf.keras.layers.SpatialDropout1D(0.2),
    tf.keras.layers.LSTM(32, dropout=0.2, recurrent_dropout=0.2, return_sequences=True),
    tf.keras.layers.LSTM(32, dropout=0.2, recurrent_dropout=0.2, return_sequences=True),
    tf.keras.layers.GlobalMaxPooling1D(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])


rlrp = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2)
model_st_lstm.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model_st_lstm.summary()


# In[ ]:


history_st_lstm = model_st_lstm.fit(train_sequences, train_labels, batch_size=32, epochs=5, 
                    validation_data=(test_sequences, test_labels), 
                    callbacks=[rlrp] ,verbose=1)


# In[ ]:


score = model_st_lstm.evaluate(test_sequences, test_labels)
print('Test Loss: ', score[0])
print('Test Accuracy', score[1])


# list all data in history
print(history_st_lstm.history.keys())
# summarize history for accuracy
plt.plot(history_st_lstm.history['accuracy'])
plt.plot(history_st_lstm.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history_st_lstm.history['loss'])
plt.plot(history_st_lstm.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# ### 7. GRU (Gated Recurrent Unit

# In[ ]:


model_gru = tf.keras.Sequential([
    tf.keras.layers.Embedding(max_words, embed_size, input_length=max_len),
    tf.keras.layers.GRU(32, dropout=0.2, recurrent_dropout=0.2, return_sequences=True),
    tf.keras.layers.GlobalMaxPooling1D(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])


rlrp = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2)
model_gru.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model_gru.summary()


# In[ ]:


history_gru = model_gru.fit(train_sequences, train_labels, batch_size=32, epochs=5, 
                    validation_data=(test_sequences, test_labels), 
                    callbacks=[rlrp] ,verbose=1)


# In[ ]:


score = model_gru.evaluate(test_sequences, test_labels)
print('Test Loss: ', score[0])
print('Test Accuracy', score[1])


# list all data in history
print(history_gru.history.keys())
# summarize history for accuracy
plt.plot(history_gru.history['accuracy'])
plt.plot(history_gru.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history_gru.history['loss'])
plt.plot(history_gru.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# ### Bidirectional LSTM

# In[ ]:


from tensorflow.keras.callbacks import ReduceLROnPlateau 

model_bidir = tf.keras.Sequential([
    tf.keras.layers.Embedding(max_words, embed_size, input_length=max_len),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
    tf.keras.layers.GlobalMaxPool1D(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])


rlrp = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2)
model_bidir.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model_bidir.summary()


# In[ ]:


history_bidir = model_bidir.fit(train_sequences, train_labels, batch_size=32, epochs=5, 
                    validation_data=(test_sequences, test_labels), 
                    callbacks=[rlrp] ,verbose=1)


# In[ ]:


score = model_bidir.evaluate(test_sequences, test_labels)
print('Test Loss: ', score[0])
print('Test Accuracy', score[1])


# list all data in history
print(history_bidir.history.keys())
# summarize history for accuracy
plt.plot(history_bidir.history['accuracy'])
plt.plot(history_bidir.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history_bidir.history['loss'])
plt.plot(history_bidir.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# ### Conclusion
# 
# The best model is using LSTM and Bidirectional along with GlobalMaxPooling and Spatial Dropout. 
# 
# Best test Accuracy - 96.71 %
# 

# ### Please Upvote the Kernel if you liked it. 
