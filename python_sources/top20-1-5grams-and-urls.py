#!/usr/bin/env python
# coding: utf-8

# # ISIS Twitter data

# Checking for top20 1-5grams as well as top20 mentions of usernames (@), tags (#) and urls.

# In[ ]:


import pandas as pd
import numpy as np
import re
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer


# In[ ]:


data = pd.read_csv('../input/tweets.csv')
data.info()


# In[ ]:


data.head()


# The features were already nicely explained by [Violinbeats - Exploring ISIS(?) Tweets](https://www.kaggle.com/violinbeats/d/kzaman/how-isis-uses-twitter/notebook-0427671092ae887aa87e).

# ## Creating function to extract different items from tweets

# Using [Marco Bonzaninis](https://marcobonzanini.com/2015/03/09/mining-twitter-data-with-python-part-2/) function for preprocessing from his "mining twitter" introduction.

# In[ ]:


regex_str = [
    r'<[^>]+>', # HTML tags
    r'(?:@[\w_]+)', # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', # URLs
 
    r'(?:(?:\d+,?)+(?:\.?\d+)?)', # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])", # words with - and '
    r'(?:[\w_]+)', # other words
    r'(?:\S)' # anything else
]
    
tokens_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)
 
def tokenize(s):
    return tokens_re.findall(s)
 
def preprocess(s, lowercase=False):
    tokens = tokenize(s)
    if lowercase:
        tokens = [token.lower() for token in tokens]
    return tokens


# # N-Grams

# Stopwords will only be removed for 1-gram but not for 2-5 grams to get the n-grams as they were written.

# ## 1-gram

# In[ ]:


vect1 = CountVectorizer(analyzer="word", stop_words="english", min_df=200, decode_error="ignore", ngram_range=(1, 1), dtype=np.int32)

# applying Vectorizer to preprocessed tweets
sub11 = vect1.fit_transform(data["tweets"].map(lambda x: " ".join(preprocess(x, lowercase = True))).tolist())

# creating (word, count) list
sub12 = zip(vect1.get_feature_names(), np.asarray(sub11.sum(axis = 0)).ravel())

# getting Top20 words
sorted(sub12, key = lambda x: x[1], reverse = True)[0:20]


# ## 2-gram

# In[ ]:


vect2 = CountVectorizer(analyzer="word", min_df=2, decode_error="ignore", ngram_range=(2, 2), dtype=np.int32)

# applying Vectorizer to preprocessed tweets
sub21 = vect2.fit_transform(data["tweets"].map(lambda x: " ".join(preprocess(x, lowercase = True))).tolist())

# creating (word, count) list
sub22 = zip(vect2.get_feature_names(), np.asarray(sub21.sum(axis = 0)).ravel())

# getting Top20 words
sorted(sub22, key = lambda x: x[1], reverse = True)[0:20]


# ## 3-gram

# In[ ]:


vect3 = CountVectorizer(analyzer="word", min_df=2, decode_error="ignore", ngram_range=(3, 3), dtype=np.int32)

# applying Vectorizer to preprocessed tweets
sub31 = vect3.fit_transform(data["tweets"].map(lambda x: " ".join(preprocess(x, lowercase = True))).tolist())

# creating (word, count) list
sub32 = zip(vect3.get_feature_names(), np.asarray(sub31.sum(axis = 0)).ravel())

# getting Top20 words
sorted(sub32, key = lambda x: x[1], reverse = True)[0:20]


# ## 4-gram

# In[ ]:


vect4 = CountVectorizer(analyzer="word", min_df=2, decode_error="ignore", ngram_range=(4, 4), dtype=np.int32)

# applying Vectorizer to preprocessed tweets
sub41 = vect4.fit_transform(data["tweets"].map(lambda x: " ".join(preprocess(x, lowercase = True))).tolist())

# creating (word, count) list
sub42 = zip(vect4.get_feature_names(), np.asarray(sub41.sum(axis = 0)).ravel())

# getting Top20 words
sorted(sub42, key = lambda x: x[1], reverse = True)[0:20]


# ## 5-gram

# In[ ]:


vect5 = CountVectorizer(analyzer="word", min_df=2, decode_error="ignore", ngram_range=(5, 5), dtype=np.int32)

# applying Vectorizer to preprocessed tweets
sub51 = vect5.fit_transform(data["tweets"].map(lambda x: " ".join(preprocess(x, lowercase = True))).tolist())

# creating (word, count) list
sub52 = zip(vect5.get_feature_names(), np.asarray(sub51.sum(axis = 0)).ravel())

# getting Top20 words
sorted(sub52, key = lambda x: x[1], reverse = True)[0:20]


# # @/usernames in tweets

# In[ ]:


tags = data["tweets"].map(lambda x: [tag for tag in preprocess(x, lowercase=True) if tag.startswith('@')])
tags = sum(tags, [])
tags[0:5]


# In[ ]:


# Top20
Counter(tags).most_common(20)


# # # in tweets

# In[ ]:


hashs = data["tweets"].map(lambda x: [hashs for hashs in preprocess(x, lowercase=True) if hashs.startswith('#')])
hashs = sum(hashs, [])
hashs[0:5]


# In[ ]:


# Top20
Counter(hashs).most_common(20)


# # urls in tweets

# In[ ]:


urls = data["tweets"].map(lambda x: [url for url in preprocess(x, lowercase=True) if url.startswith('http:') or url.startswith('https:')])
urls = sum(urls, [])
urls[0:5]


# In[ ]:


# Top20
Counter(urls).most_common(20)


# Because of the limited number of characters in a tweet it looks like some urls are separated over multiple tweets.
