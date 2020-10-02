#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import numpy as np
from gensim.models import word2vec
from gensim.models import FastText
from sklearn.manifold import TSNE
from nltk.cluster import KMeansClusterer
import nltk
from sklearn import metrics
from nltk.cluster import KMeansClusterer
import nltk
import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn.cluster import KMeans
import re
from string import punctuation
from nltk.corpus import stopwords 
stop_words = set(stopwords.words('english')) 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


train=pd.read_csv("../input/tweet-sentiment-extraction/train.csv")
test=pd.read_csv("../input/tweet-sentiment-extraction/test.csv")
sample=pd.read_csv("../input/tweet-sentiment-extraction/sample_submission.csv")


# In[ ]:


train.shape


# In[ ]:


train.sentiment.value_counts()


# In[ ]:


def text_to_wordlist(text, remove_stop_words=True, stem_words=False,lemmatize_words=False):
    # Clean the text, with the option to remove stop_words and to stem words.
    # Clean the text
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
#     text = re.sub(r"</p>", "", text)
#     text = re.sub(r"<p>", "", text)
    text = re.sub(r"what's", "", text)
    text = re.sub(r"\n", "", text)
    text = re.sub(r"What's", "", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"I'm", "I am", text)
    text = re.sub(r" m ", " am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r" J K ", " JK ", text)
    # Remove punctuation from text
    text = ''.join([c.lower() for c in text if c not in punctuation])

    # Optionally, remove stop words
    if remove_stop_words:
        text = text.split()
        text = [w for w in text if not w in stop_words]
        text = " ".join(text)

    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)

    if lemmatize_words:
        # stemmed_words = [lemmatizer.lemmatize(word) for word in text]
        # text = " ".join(stemmed_words)
        pass
    # Return a list of words
    return (text)
# text_to_wordlist(stack['body'])


# In[ ]:


for i,val in enumerate(train['text']):
    try:
        train['text'][i]=text_to_wordlist(train['text'][i])
    except:
        continue
# train['text']    
# text_to_wordlist(train['text'][2])


# In[ ]:


train['text']


# In[ ]:


corpus=[]
for i in train['text'].values:
    corpus.append(str(i).split(" "))
corpus[:1]


# In[ ]:



model = FastText(corpus, size=100, workers=4,window=5)


# In[ ]:


print(model.wv.most_similar('shit'))
print('******')
print(model.wv.most_similar('crap'))
print('******')
print(model.wv.most_similar('good'))


# In[ ]:


model = word2vec.Word2Vec(corpus, size=100, workers=4,window=5)


# In[ ]:


# print(model.wv.most_similar('shit')) gives error because shit word not present in vocabulary
print('******')
# print(model.wv.most_similar('crap'))  gives error because crap word not present in vocabulary
print('******')
print(model.wv.most_similar('good'))


# what do you guys think ?
# 
# going forward i'll probably use these embeddings for classfication , until then search is on . if you like this give it a thumbs up . would like to collaborate with everyone of ya !!

# In[ ]:




