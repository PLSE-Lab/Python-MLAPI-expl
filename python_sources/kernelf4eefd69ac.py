#!/usr/bin/env python
# coding: utf-8

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

# Any results you write to the current directory are saved as output.


# In[ ]:


from __future__ import print_function

from keras.preprocessing import sequence
from keras import backend as K
from keras.engine.topology import Layer
from keras.models import Model
from keras.layers import *
from gensim.models.word2vec import Word2Vec
import multiprocessing
from nltk.stem.lancaster import LancasterStemmer
from nltk.tokenize import RegexpTokenizer
import re


# In[ ]:





# In[ ]:


def import_tweets(filename, header = None):
    #import data from csv file via pandas library
    tweet_dataset = pd.read_csv(filename, encoding = 'latin-1', header = header)
    #the column names are based on sentiment140 dataset provided on kaggle
    tweet_dataset.columns = ['sentiment','id','date','flag','user','text']
    #delete 3 columns: flags,id,user, as they are not required for analysis
    for i in ['flag','id','user','date']: del tweet_dataset[i] # or tweet_dataset = tweet_dataset.drop(["id","user","date","user"], axis = 1)
    #in sentiment140 dataset, positive = 4, negative = 0; So we change positive to 1
    tweet_dataset.sentiment = tweet_dataset.sentiment.replace(4,1)
    return tweet_dataset


# In[ ]:


tweet_dataset=import_tweets("../input/training.1600000.processed.noemoticon.csv")


# In[ ]:


def handle_emojis(tweet):
    # Smile -- :), : ), :-), (:, ( :, (-:, :')
    tweet = re.sub(r'(:\s?\)|:-\)|\(\s?:|\(-:|:\'\))', ' EMO_POS ', tweet)
    # Laugh -- :D, : D, :-D, xD, x-D, XD, X-D
    tweet = re.sub(r'(:\s?D|:-D|x-?D|X-?D)', ' EMO_POS ', tweet)
    # Love -- <3, :*
    tweet = re.sub(r'(<3|:\*)', ' EMO_POS ', tweet)
    # Wink -- ;-), ;), ;-D, ;D, (;,  (-;
    tweet = re.sub(r'(;-?\)|;-?D|\(-?;)', ' EMO_POS ', tweet)
    # Sad -- :-(, : (, :(, ):, )-:
    tweet = re.sub(r'(:\s?\(|:-\(|\)\s?:|\)-:)', ' EMO_NEG ', tweet)
    # Cry -- :,(, :'(, :"(
    tweet = re.sub(r'(:,\(|:\'\(|:"\()', ' EMO_NEG ', tweet)
    return tweet


# In[ ]:


def preprocess_tweet(tweet):
    #Preprocess the text in a single tweet
    #arguments: tweet = a single tweet in form of string 
    #convert the tweet to lower case
    tweet = tweet.lower()
    #convert all urls to sting "URL"
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)
    #convert all @username to "AT_USER"
    tweet = re.sub('@[^\s]+','AT_USER', tweet)
    #convert "#topic" to just "topic"
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    # Remove RT (retweet)
    tweet = re.sub(r'\brt\b', '', tweet)
    # Convert more than 2 letter repetitions to 2 letter
    # funnnnny --> funny
    tweet = re.sub(r'(.)\1+', r'\1\1', tweet)
    # Replace 2+ dots with space
    tweet = re.sub(r'\.{2,}', ' ', tweet)
    # Strip space, " and ' from tweet
    tweet = tweet.strip(' "\'')
    #Replace occurences of mentioning @UserNames
    tweet = tweet.replace('@\w+', ' ')
    #Replace links contained in the tweet
    tweet = tweet.replace('http\S+', ' ')
    tweet = tweet.replace('www.[^ ]+', ' ')
    #remove numbers
    tweet = tweet.replace('[0-9]+', ' ')
    #replace special characters and puntuation marks
    tweet = re.sub(r'[!"#$%&()*+,-./:;<=>?@[\]^_`{|}~]', ' ', tweet)
    # Replace emojis with either EMO_POS or EMO_NEG
    tweet = handle_emojis(tweet)
    #correct all multiple white spaces to a single white space
    tweet = re.sub('[\s]+', ' ', tweet)
    return tweet


# In[ ]:


tweet_dataset['text'] = tweet_dataset['text'].apply(handle_emojis)


# In[ ]:


tweet_dataset['text'] = tweet_dataset['text'].apply(preprocess_tweet)


# In[ ]:


tweet_dataset['text'].head()


# In[ ]:


tweet_dataset[tweet_dataset.sentiment ==1]


# In[ ]:


print(tweet_dataset[ tweet_dataset['sentiment'] == 1].size)
print(tweet_dataset[ tweet_dataset['sentiment'] == 0].size)
print(len(tweet_dataset))


# In[ ]:


labels=tweet_dataset['sentiment']
corpus = tweet_dataset['text']


# In[ ]:


#writing the dataframe to csv file
lables = tweet_dataset['sentiment'].to_csv('sentixData_lables')
corpus = tweet_dataset['text'].to_csv('sentixData_corpus')


# In[ ]:


# Loading the data
lables = pd.read_csv('sentixData_lables', header=None)
corpus = pd.read_csv('sentixData_corpus', header=None)


# In[ ]:


corpus.head()


# In[ ]:


print('Corpus size: {}'.format(len(corpus)))


# In[ ]:


type(corpus)


# In[ ]:


corpus = corpus.to_string()


# ## Tokenize and Stem

# In[ ]:


# Tokenize and stem
tkr = RegexpTokenizer('[a-zA-Z0-9@]+')
stemmer = LancasterStemmer()

tokenized_corpus = []
for i, tweet in enumerate(corpus):
    tokens = [stemmer.stem(t) for t in tkr.tokenize(tweet) if not t.startswith('@')]
    tokenized_corpus.append(tokens)


# In[ ]:


# saving
with open('tokenized_corpus.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


# In[1]:


# loading
with open('tokenized_corpus.pickle', 'rb') as handle:
    tokenized_corpus = pickle.load(handle)


# In[ ]:


# # Gensim Word2Vec model
# vector_size = 512
# window_size = 10


# In[ ]:


# # Create Word2Vec
# word2vec = Word2Vec(sentences=tokenized_corpus,
#                     size=vector_size, 
#                     window=window_size, 
#                     negative=20,
#                     iter=50,
#                     seed=1000,
#                     workers=multiprocessing.cpu_count())


# In[ ]:


# # Copy word vectors and delete Word2Vec model  and original corpus to save memory
# X_vecs = word2vec.wv
# del word2vec
# del corpus


# In[ ]:




