#!/usr/bin/env python
# coding: utf-8

# # FAKE NEWS CLASSIFICATION USING DEEP LEARNING WITH GloVe
# 
# 
# In this notebook i have tried to classify news into 2 classes real and fake using LSTM nueral network .
# 
# ![](https://cdn.factcheck.org/UploadedFiles/fakenews.jpg)
# I have used pretrained Glove for vectorization and able to achive an accuracy of 99% by the proposed LSTM model.
# 

# ## Dataset
# The dataset consists of about 40000 articles consisting around equal number of fake as well as real news Most of the news where collected from U.S newspapers and contian news about american poltics,world news ,news etc.

# # Loading the dataset

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
""
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style("darkgrid")


# In[ ]:


true = pd.read_csv("../input/fake-and-real-news-dataset/True.csv")
false = pd.read_csv("../input/fake-and-real-news-dataset/Fake.csv")


# * Finding the most used words in fake and real news using Word cloud

# In[ ]:


from wordcloud import WordCloud, ImageColorGenerator
text = " ".join(str(each) for each in true.text.unique())
wordcloud = WordCloud(max_words=200,colormap='Set3', background_color="black").generate(text)
plt.figure(figsize=(15,10))
plt.imshow(wordcloud, interpolation='Bilinear')
plt.axis("off")
plt.figure(1,figsize=(12, 12))
plt.show()


# In[ ]:


from wordcloud import WordCloud, ImageColorGenerator
text = " ".join(str(each) for each in false.text.unique())
wordcloud = WordCloud(max_words=200,colormap='Set3', background_color="black").generate(text)
plt.figure(figsize=(15,10))
plt.imshow(wordcloud, interpolation='Bilinear')
plt.axis("off")
plt.figure(1,figsize=(12, 12))
plt.show()


# * labeling the fake news as 0 and real news as 1

# In[ ]:


true['label'] = 1
false['label'] = 0


# * Merging the 2 datasets

# In[ ]:


news = pd.concat([true,false]) 
news


# In[ ]:


news['text'] = news['text'] + " " + news['title']
news


# In[ ]:


df=news.drop(["date","title","subject"],axis=1)


# In[ ]:


df


# * containing  23481 fake news and 21417 non fake news

# In[ ]:


print(false.shape)
print(true.shape)


# In[ ]:


sns.countplot(x="label", data=news);
plt.show()


# # data Preproccessing
# We have to convert the raw messages (sequence of characters) into vectors (sequences of numbers).before that we need to do the following:
# 1. Remove punctuation
# 2. Remove numbers
# 3. Remove tags
# 4. Remove urls
# 5. Remove stepwords
# 6. Change the news to lower case
# 7. Lemmatisation 

# In[ ]:


import nltk
import string
from nltk.corpus import stopwords
import re


# The following 4 functions will help as to remove punctions (<,.'':, etc),numbers,tags and urls

# In[ ]:


def rem_punctuation(text):
  return text.translate(str.maketrans('','',string.punctuation))

def rem_numbers(text):
  return re.sub('[0-9]+','',text)


def rem_urls(text):
  return re.sub('https?:\S+','',text)


def rem_tags(text):
  return re.sub('<.*?>'," ",text)


# In[ ]:


df['text'].apply(rem_urls)
df['text'].apply(rem_punctuation)
df['text'].apply(rem_tags)
df['text'].apply(rem_numbers)


# rem_stopwords() is the function for removing stopwords and for converting the words to lower case

# In[ ]:


stop = set(stopwords.words('english'))

def rem_stopwords(df_news):
    
    words = [ch for ch in df_news if ch not in stop]
    words= "".join(words).split()
    words= [words.lower() for words in df_news.split()]
    
    return words    


# In[ ]:


df['text'].apply(rem_stopwords)


# * **Lemmatization** 
# performs vocabulary and morphological analysis of the word and is normally aimed at removing **inflectional endings** only.That isconvert the words to their base or root form eg in "plays" it is converted to "play" by removing "s"

# In[ ]:


from nltk.stem import WordNetLemmatizer
#nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

def lemmatize_words(text):
  lemmas = []
  for word in text.split():
    lemmas.append(lemmatizer.lemmatize(word))
  return " ".join(lemmas)


# In[ ]:


df['text'].apply(lemmatize_words)


# # Tokenizing & Padding
# 
# * **Tokenizing**
# is the process of breaking down a text into words. Tokenization can happen on any character, however the most common way of tokenization is to do it on space character.
# 
# * **Padding**
# Naturally, some of the sentences are longer or shorter. We need to have the inputs with the same size, for this we use padding

# In[ ]:


from keras.preprocessing.text import Tokenizer

from keras.utils import to_categorical


# In[ ]:


x = df['text'].values
y= df['label'].values


# In[ ]:


tokenizer = Tokenizer()
tokenizer.fit_on_texts(x)
word_to_index = tokenizer.word_index
x = tokenizer.texts_to_sequences(x)


# In[ ]:


vocab_size =  len(word_to_index)
oov_tok = "<OOV>"
max_length = 250
embedding_dim = 100


# In[ ]:


from keras.preprocessing.sequence import pad_sequences

x = pad_sequences(x, maxlen=max_length)


# # Vectorization
#  Word vectorization is a methodology in NLP to map words or phrases from vocabulary to a corresponding vector of real numbers 
#  There are many  method for doing vectorization including  Bag of words,TFIDF or prettrained method such as Word2Vec ,Glove etc
#  
#  we are using **GloVe** learning algorithm for obtaining vector representations for words devolped by Stanford
#  

# In[ ]:


embeddings_index = {};
with open('../input/glove6b100dtxt/glove.6B.100d.txt') as f:
    for line in f:
        values = line.split();
        word = values[0];
        coefs = np.asarray(values[1:], dtype='float32');
        embeddings_index[word] = coefs;

embeddings_matrix = np.zeros((vocab_size+1, embedding_dim));
for word, i in word_to_index.items():
    embedding_vector = embeddings_index.get(word);
    if embedding_vector is not None:
        embeddings_matrix[i] = embedding_vector;


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=1)


# In[ ]:


import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size+1, embedding_dim, input_length=max_length, weights=[embeddings_matrix], trainable=False),
    tf.keras.layers.LSTM(64,return_sequences=True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')

   
])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()


# In[ ]:


epochs = 10
history = model.fit(X_train,y_train,epochs=epochs,validation_data=(X_test,y_test),batch_size=128)


# In[ ]:


result = model.evaluate(X_test, y_test)
# extract those
loss = result[0]
accuracy = result[1]


print(f"[+] Accuracy: {accuracy*100:.2f}%")

