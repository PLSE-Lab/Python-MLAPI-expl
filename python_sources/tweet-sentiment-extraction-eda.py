#!/usr/bin/env python
# coding: utf-8

# # Tweet Sentiment Extraction - EDA

# <a class="anchor" id="0.1"></a>
# ## Table of Contents
#   
# 1. [Import libraries](#1)
# 1. [Download data](#2)
# 1. [EDA](#3)
# 1. [WordCloud](#4)

# ## 1. Import libraries <a class="anchor" id="1"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


import nltk
#nltk.download('stopwords')


# In[ ]:


import pandas as pd
import numpy as np
import os

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from nltk.corpus import stopwords
from nltk.util import ngrams
stop = set(stopwords.words('english'))

from wordcloud import WordCloud

from collections import defaultdict
from collections import Counter
plt.style.use('ggplot')

from sklearn.feature_extraction.text import CountVectorizer
import string


# ## 2. Download data <a class="anchor" id="2"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


os.listdir('../input/tweet-sentiment-extraction')


# In[ ]:


train = pd.read_csv('../input/tweet-sentiment-extraction/train.csv')
test = pd.read_csv('../input/tweet-sentiment-extraction/test.csv')
submission = pd.read_csv("../input/tweet-sentiment-extraction/sample_submission.csv")


# In[ ]:


print('There are {} rows and {} columns in train'.format(train.shape[0],train.shape[1]))
print('There are {} rows and {} columns in test'.format(test.shape[0],test.shape[1]))


# In[ ]:


train.head(10)


# In[ ]:


train.info()


# It seems that `text` and `selected_text` got one missing data.

# In[ ]:


train[train.text.isnull()]


# Let's remove it.

# In[ ]:


train = train[train['text'].notna()]


# ## 3. EDA <a class="anchor" id="3"></a>
# 
# [Back to Table of Contents](#0.1)

# ### Class distribution

# In[ ]:


train.sentiment.unique()


# In[ ]:


positive_len = train[train['sentiment']=='positive'].shape[0]
neutral_len = train[train['sentiment']=='neutral'].shape[0]
negative_len = train[train['sentiment']=='negative'].shape[0]


# In[ ]:


plt.rcParams['figure.figsize'] = (7, 5)
labels = ['Positive', 'asd0', 'bas']
plt.bar(10,positive_len,3, label="Positive", color='green')
plt.bar(15,neutral_len,3, label="Neutral", color='gray')
plt.bar(20,negative_len,3, label="Negative", color='red')
plt.legend()
plt.ylabel('Number of examples')
plt.title('Sentiment distribution')
plt.show()


# ### Number of characters in tweets

# In[ ]:


def length(text):    
    '''a function which returns the length of text'''
    return len(text)


# In[ ]:


train['length'] = train['text'].apply(length)


# In[ ]:


plt.rcParams['figure.figsize'] = (18.0, 6.0)
bins = 150
plt.hist(train[train['sentiment']=='positive']['length'], alpha=0.3, bins=bins, label='Positive')
plt.hist(train[train['sentiment']=='neutral']['length'], alpha=0.5, bins=bins, label='Neutral')
plt.hist(train[train['sentiment']=='negative']['length'], alpha=0.65, bins=bins, label='Negative')
plt.xlabel('length')
plt.ylabel('numbers')
plt.legend(loc='upper right')
plt.xlim(0,150)
plt.grid()
plt.show()


# In[ ]:


fig,(ax1,ax2,ax3)=plt.subplots(1,3,figsize=(15,5))
tweet_len=train[train['sentiment']=='positive']['text'].str.len()
ax1.hist(tweet_len,color='green')
ax1.set_title('positive tweets')

tweet_len=train[train['sentiment']=='neutral']['text'].str.len()
ax2.hist(tweet_len,color='gray')
ax2.set_title('neutral tweets')

tweet_len=train[train['sentiment']=='negative']['text'].str.len()
ax3.hist(tweet_len,color='red')
ax3.set_title('negative tweets')

fig.suptitle('Characters in tweets')
plt.show()


# ### Number of words in a tweet

# In[ ]:


fig,(ax1,ax2,ax3)=plt.subplots(1,3,figsize=(15,5))
tweet_len=train[train['sentiment']=='positive']['text'].str.split().map(lambda x: len(x))
ax1.hist(tweet_len,color='green')
ax1.set_title('positive tweets')

tweet_len=train[train['sentiment']=='neutral']['text'].str.split().map(lambda x: len(x))
ax2.hist(tweet_len,color='gray')
ax2.set_title('neutral tweets')

tweet_len=train[train['sentiment']=='negative']['text'].str.split().map(lambda x: len(x))
ax3.hist(tweet_len,color='red')
ax3.set_title('negative tweets')

fig.suptitle('Words in a tweet')
plt.show()


# ###  Average word length in a tweet

# In[ ]:


fig,(ax1,ax2,ax3) = plt.subplots(1,3,figsize=(15,5))

word=train[train['sentiment']=='positive']['text'].str.split().apply(lambda x : [len(i) for i in x])
sns.distplot(word.map(lambda x: np.mean(x)),ax=ax1,color='green')
ax1.set_title('positive tweets')

word=train[train['sentiment']=='neutral']['text'].str.split().apply(lambda x : [len(i) for i in x])
sns.distplot(word.map(lambda x: np.mean(x)),ax=ax2,color='gray')
ax2.set_title('neutral tweets')

word=train[train['sentiment']=='negative']['text'].str.split().apply(lambda x : [len(i) for i in x])
sns.distplot(word.map(lambda x: np.mean(x)),ax=ax3,color='red')
ax3.set_title('negative tweets')

fig.suptitle('Average word length in each tweet')
plt.show()


# In[ ]:


def create_corpus(target):
    corpus = []
    
    for x in train[train['sentiment']==target]['text'].str.split():
        for i in x:
            corpus.append(i)
    return corpus


# ### Common stopwords in tweets

# ##### `Positive` class

# In[ ]:


corpus = create_corpus('positive')

dic = defaultdict(int)
for word in corpus:
    if word in stop:
        dic[word]+=1
        
top = sorted(dic.items(), key=lambda x:x[1],reverse=True)[:10]


# In[ ]:


plt.rcParams['figure.figsize'] = (18.0, 6.0)
x,y = zip(*top)
plt.bar(x,y,color='green')
plt.show()


# ##### `Neutral` class

# In[ ]:


corpus = create_corpus('neutral')

dic = defaultdict(int)
for word in corpus:
    if word in stop:
        dic[word]+=1

top = sorted(dic.items(), key=lambda x:x[1],reverse=True)[:10] 
    

plt.rcParams['figure.figsize'] = (18.0, 6.0)
x,y = zip(*top)
plt.bar(x,y,color='gray')
plt.show()


# ##### `Negative` class

# In[ ]:


corpus = create_corpus('negative')

dic = defaultdict(int)
for word in corpus:
    if word in stop:
        dic[word]+=1

top = sorted(dic.items(), key=lambda x:x[1],reverse=True)[:10] 
    

plt.rcParams['figure.figsize'] = (18.0, 6.0)
x,y = zip(*top)
plt.bar(x,y,color='red')
plt.show()


# In all the tweets belonging to the three sentiment classes `to` is the most common stopword, followed by `the` and `a` (except for the *Negative class* that has `my` before `a`).

# ### Analyzing punctuations

# ##### `Positive` class

# In[ ]:


plt.figure(figsize=(16,5))
corpus = create_corpus('positive')

dic = defaultdict(int)
special = string.punctuation
for i in (corpus):
    if i in special:
        dic[i]+=1
        
x,y = zip(*dic.items())
plt.bar(x,y,color='green')
plt.show()


# ##### `Neutral` class

# In[ ]:


plt.figure(figsize=(16,5))
corpus=create_corpus('neutral')
dic = defaultdict(int)
special = string.punctuation
for i in (corpus):
    if i in special:
        dic[i]+=1
        
x,y = zip(*dic.items())
plt.bar(x,y,color='gray')
plt.show()


# ##### `Negative` class

# In[ ]:


plt.figure(figsize=(16,5))
corpus = create_corpus('negative')
dic = defaultdict(int)
special = string.punctuation
for i in (corpus):
    if i in special:
        dic[i]+=1
        
x,y = zip(*dic.items())
plt.bar(x,y,color='red')
plt.show()


# ### Common words

# In[ ]:


plt.figure(figsize=(16,5))
counter = Counter(corpus)
most = counter.most_common()
x = []
y = []
for word,count in most[:40]:
    if (word not in stop) :
        x.append(word)
        y.append(count)


# In[ ]:


sns.barplot(x=y,y=x)
plt.show()


# ### N-gram analysis

# In[ ]:


def get_top_tweet_ngrams(corpus, ngram=2, n=None):
    vec = CountVectorizer(ngram_range=(ngram, ngram)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]


# ##### Most common bigrams

# In[ ]:


plt.figure(figsize=(16,5))
top_tweet_bigrams=get_top_tweet_ngrams(train['text'], 2)[:10]
x,y=map(list,zip(*top_tweet_bigrams))
sns.barplot(x=y,y=x)
plt.show()


# ##### Most common trigrams

# In[ ]:


plt.figure(figsize=(16,5))
top_tweet_bigrams=get_top_tweet_ngrams(train['text'], 3)[:10]
x,y=map(list,zip(*top_tweet_bigrams))
sns.barplot(x=y,y=x)
plt.show()


# ## 4. WordCloud <a class="anchor" id="4"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


def create_corpus_df(tweet, target):
    corpus = []
    
    for x in tweet[tweet['sentiment']==target]['text'].str.split():
        for i in x:
            corpus.append(i)
    return corpus


# ##### `Positive` class

# In[ ]:


corpus_new_positive = create_corpus_df(train,'positive')
len(corpus_new_positive)


# In[ ]:


corpus_new_positive[:10]


# In[ ]:


plt.figure(figsize=(12,8))
word_cloud = WordCloud(
    background_color='black',
    max_font_size=80
).generate(" ".join(corpus_new_positive[:50]))
plt.imshow(word_cloud)
plt.axis('off')
plt.show()


# ##### `Neutral` class

# In[ ]:


corpus_new_neutral = create_corpus_df(train,'neutral')
len(corpus_new_neutral)


# In[ ]:


corpus_new_neutral[:10]


# In[ ]:


# Generating the wordcloud with the values under the category dataframe
plt.figure(figsize=(12,8))
word_cloud = WordCloud(
    background_color='black',
    max_font_size=80
).generate(" ".join(corpus_new_neutral[:50]))
plt.imshow(word_cloud)
plt.axis('off')
plt.show()


# ##### `Negative` class

# In[ ]:


corpus_new_negative = create_corpus_df(train,'negative')
len(corpus_new_negative)


# In[ ]:


corpus_new_negative[:10]


# In[ ]:


plt.figure(figsize=(12,8))
word_cloud = WordCloud(
    background_color='black',
    max_font_size=80
).generate(" ".join(corpus_new_negative[:50]))
plt.imshow(word_cloud)
plt.axis('off')
plt.show()

