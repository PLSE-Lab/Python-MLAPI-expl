#!/usr/bin/env python
# coding: utf-8

# # Introduction
# in a [previous kernel](https://www.kaggle.com/nicodds/rome-wasn-t-built-in-a-day-spotting-fake-reviews), I performed a simple outlier analysis on [Rome B&Bs reviews dataset](https://www.kaggle.com/nicodds/rome-b-and-bs). This revealed that users writing 9 or more reviews could be considered as fake users. Furthermore, almost all such users share the same username pattern (a name starting with a capital letter, a space and a single capital letter).
# 
# Since appetite comes with eating, I propose to further improve the estimate using Natural Language Processing (NLP) techniques on the review contents. In particular, I'll extract some lexical and punctuation features to show, using cluster analysis, that these texts belong to the same author.
# 
# This is my first try with NLP and clustering, please upvote the kernel if you like it or give me suggestions to improve it.
# 
# ## The code

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import string
import re


# For the sake of simplicity, I'll focus only on english reviews, that correspond more or less to an half of the suspect reviews. A similar analysis has been performed also on reviews written in italian (corresponding almost to the other half), yielding the similar results.

# In[2]:


language = 'english'
sentence_tokenizer = nltk.data.load('tokenizers/punkt/%s.pickle' % language)
stop_words=set(nltk.corpus.stopwords.words(language))


# I'm defining here some utility functions to ease the process of feature extraction from text

# In[3]:


def clean_tokenize(text):
    text = text.lower().replace('b&b', 'bb')
    tmp_tokens = nltk.tokenize.word_tokenize(text)
    no_punctuation = []
    x=re.compile('^[%s]+$' % re.escape(string.punctuation))
    
    for tk in tmp_tokens:
        if not x.match(tk):
            no_punctuation.append(tk)

    return [token for token in no_punctuation if token not in stop_words]

def feature_extractor(text):
    # we don't consider any stop_words
    capital_lett_cnt = len(re.findall(r'[A-Z]', text))
    meaningful_words = clean_tokenize(text)
    vocabulary       = set(meaningful_words)
    sentences        = sentence_tokenizer.tokenize(text)
    sentences_number = float(len(sentences))
    # wps stands for word per sentence
    meaningful_wps   = np.array([len(clean_tokenize(s)) for s in sentences])
    
    # we return a list of features ordered as:
    # 1: mean meaningful_wps
    # 2: std_dev meaningful_wps
    # 3: lexical diversity index (:= len(vocabulary) / len(words)), which
    #    accounts the lexical richness of the text
    # 4: commas per sentence
    # 5: semicolons per sentence
    # 6: colons per sentence 
    # 7: exclamations per sentence (it should be <= 1)
    # 8: capital letters per sentence (it should be >= 1)
    return meaningful_wps.mean(),        meaningful_wps.std(),        len(vocabulary) / float(len(meaningful_words)),        text.count(',') / sentences_number,         text.count(';') / sentences_number,         text.count(':') / sentences_number,         text.count('!') / sentences_number,         capital_lett_cnt


# In[4]:


reviews = pd.read_csv('../input/reviews.csv', encoding='utf-8')
reviews['review_text'] = reviews['review_text'].str.replace('...More$', '', case=False)


# Let's select only the text of the reviews, obtaining a corpus of more than 26k texts

# In[5]:


pattern = '^[A-Z][a-z]+ [A-Z]$'
suspect_en = reviews[(reviews['review_user'].str.match(pattern)) & (reviews['review_language'] == 'en')].review_text
len(suspect_en)


# I'll create now a features vector that will contain all the features extracted from the previous texts (I know that iterating over a pandas series is ugly, but I have not found a better way).

# In[6]:


scaler = StandardScaler()
features_vector = np.ndarray((len(suspect_en), 8))
for i, text in enumerate(suspect_en):
    features_vector[i] = feature_extractor(text)
features_vector = scaler.fit_transform(features_vector)


# I'm not aware of the number of single authors amog the suspect. Furthermore, some users could be real. So I decided to perform cluster analysis using [DBSCAN algorithm](https://en.wikipedia.org/wiki/DBSCAN). In particular, I'm setting to 9 (i.e. number of features + 1) the minimum number of cluster members. For the threshold value of the distance, I'm resorting to scikit-learn default.

# In[7]:


db = DBSCAN(min_samples=9)
y_pred = db.fit_predict(features_vector)

# Number of clusters in labels, ignoring noise if present.
n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)

print('Estimated number of clusters: %d (noise: %d)' % (n_clusters, np.count_nonzero(db.labels_==-1)))


# In[8]:


plt.subplots(figsize=(13,5))
sns.countplot(x=db.labels_[2:])
plt.xticks(rotation=90)
plt.show()


# In[9]:


np.count_nonzero(db.labels_==0)


# We see that a large share of the selected reviews are in the same cluster label. This means that they share common text features. How many reviews are in the other clusters?

# In[10]:


plt.subplots(figsize=(13,5))
sns.countplot(x=db.labels_[db.labels_>0])
plt.xticks(rotation=90)
plt.show()


# As it is easy to see, the other clusters are definitely less populated. Only three clusters show a number of reviews greater than 30.

# ## Conclusions
# Using cluster analysis, I have shown that a large share of the reviews written in English by users with the same username pattern have common text characteristics. Using a similar technique, the same result is also found for Italian reviews.
# 
# This confirms and extends the results obtained in the previous kernel using the outlier analysis. In fact, this discovery implies that  near 36k (out of 223k) reviews of Rome B&Bs should be considered suspicious. Using data from [TripAdvisor](http://www.tripadvisor.com) databases, this suspicion could be confirmed or removed. For example, by looking at the details of users' Internet connections, you can easily identify a common place of origin in the details of various suspicious users.

# In[ ]:




