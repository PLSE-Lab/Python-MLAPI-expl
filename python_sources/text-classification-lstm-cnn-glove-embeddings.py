#!/usr/bin/env python
# coding: utf-8

# This kernel is based on the tutorial series written by Sabber Ahamed @msahamed on the Medium.
# https://medium.com/@sabber/classifying-yelp-review-comments-using-cnn-lstm-and-pre-trained-glove-word-embeddings-part-3-53fcea9a17fa

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
# Keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
from keras.layers.embeddings import Embedding
# Any results you write to the current directory are saved as output.
# Others
import nltk
import string
from nltk.corpus import stopwords
import re
from nltk.stem.snowball import SnowballStemmer


# In[ ]:


train_df=pd.read_csv('../input/train.csv')


# In[ ]:


train_df.shape


# #### Text Preprocessing

# In[ ]:


def clean_text(text):
    
    ## Remove puncuation
    text = text.translate(string.punctuation)
    
    ## Convert words to lower case and split them
    text = text.lower().split()
    
    ## Remove stop words
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops and len(w) >= 3]
    
    text = " ".join(text)

    ## Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    ## Stemming
    text = text.split()
    stemmer = SnowballStemmer('english')
    stemmed_words = [stemmer.stem(word) for word in text]
    text = " ".join(stemmed_words)

    return text


# In[ ]:


train_df['question_text'] = train_df['question_text'].map(lambda x: clean_text(x))


# #### Maximum length of the question text is set to be 100. If the text is less 100 characters, zeroes will be padded else the text will be truncated

# In[ ]:


tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_df['question_text'])

train_sequences = tokenizer.texts_to_sequences(train_df['question_text'])
train_data = pad_sequences(train_sequences, maxlen=100)


# #### Using Glove pre-trained model

# In[ ]:



EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))


# In[ ]:


# create a weight matrix for words in training docs
embedding_matrix = np.zeros((50000, 300))
for word, index in tokenizer.word_index.items():
    if index > 50000 - 1:
        break
    else:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector


# #### Embedding vector forms the first layer followed by Convolutional network and finally wrapped by LSTM.

# In[ ]:


model_glove = Sequential()
model_glove.add(Embedding(50000, 300, input_length=100, weights=[embedding_matrix], trainable=False))
model_glove.add(Dropout(0.2))
model_glove.add(Conv1D(64, 5, activation='relu'))
model_glove.add(MaxPooling1D(pool_size=4))
model_glove.add(LSTM(300))
model_glove.add(Dense(1, activation='sigmoid'))
model_glove.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[ ]:


model_glove.fit(train_data, train_df['target'], validation_split=0.2, epochs = 2)


# #### It takes around 1 hour to fit the model on the data for 2 epochs. Please note that I haven't split the training data into train and valid sets for simplicity.

# In[ ]:


test_df=pd.read_csv('../input/test.csv')


# #### Pre-processing on the text data.

# In[ ]:


test_df['question_text'] = test_df['question_text'].map(lambda x: clean_text(x))


# In[ ]:


test_sequences = tokenizer.texts_to_sequences(test_df['question_text'])
test_data = pad_sequences(test_sequences, maxlen=100)


# In[ ]:


predictions= model_glove.predict_classes(test_data)


# In[ ]:


submission_df = pd.DataFrame({"qid":test_df["qid"].values})
submission_df['prediction'] = predictions
submission_df.to_csv("submission.csv", index=False)


# 

# In[ ]:




