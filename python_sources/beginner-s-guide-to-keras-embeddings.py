#!/usr/bin/env python
# coding: utf-8

# acknowledgements :- https://www.kaggle.com/tanulsingh077/twitter-sentiment-extaction-analysis-eda-and-model,
# https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import re
import string
import numpy as np 
import random
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from collections import Counter
from pandas_profiling import ProfileReport 
from wordcloud import  STOPWORDS
from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
import nltk
from nltk.corpus import stopwords

from tqdm import tqdm
import os
import nltk
import random

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


train = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')
test = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')
ss = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/sample_submission.csv')


# In[ ]:


print(train.shape)
print(test.shape)


# In[ ]:


report = ProfileReport(train)
report


# In[ ]:


train.dropna(inplace=True)


# In[ ]:


train.head()


# In[ ]:


train.sentiment.value_counts(normalize=True)*100


# most comments are neutral followed by positive and then negative..

# In[ ]:


def clean_text(text):
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text


# In[ ]:


train['text'] = train['text'].apply(lambda x:clean_text(x))
train['selected_text'] = train['selected_text'].apply(lambda x:clean_text(x))


# In[ ]:


train.head()


# In[ ]:


train['text'].values


# In[ ]:


# integer encode the documents
vocab_size = 10000
encoded_docs = [one_hot(d, vocab_size) for d in train['selected_text'].values]
print(encoded_docs[0])


# In[ ]:


# pad documents to a max length of 4 words
max_length = 4+6
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='pre')
print(padded_docs[0])


# In[ ]:


# define the model
model = Sequential()
model.add(Embedding(vocab_size, 8, input_length=max_length))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# summarize the model
print(model.summary())


# In[ ]:


print(model.predict(padded_docs))


# In[ ]:


print(model.predict(padded_docs)[0])


# since we can see that words have now been converted to vectors , we can use this representation for prediction purposes . this will be the goal for next notebook ..

# 

# In[ ]:




