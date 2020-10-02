#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
from time import time

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

for dirname, _, filenames in os.walk('../input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd


# In[ ]:


articles = pd.read_csv('../input/medium-articles/articles.csv')


# In[ ]:


articles.head()


# In[ ]:


articles.info()


# # Pre-processing

# In[ ]:


# drop useless columns and rows
articles = articles.drop('link', axis=1)                   .drop_duplicates()                   .reset_index(drop=True)

# convert claps to floats
articles['claps'] = articles['claps'].apply(lambda s: float(s) if s[-1] != 'K' else float(s[:-1])*1000)

# add author library count
author_count = pd.value_counts(articles['author'])
articles['author_lib_count'] = articles['author'].apply(lambda a : author_count[a])
articles.drop('author', axis=1, inplace=True)


# In[ ]:


articles


# # Analysis

# ## Total number of words

# In[ ]:


sum(len(article.split(" ")) for article in articles["text"])


# ## Authors' articles count

# In[ ]:


pd.value_counts(articles['author_lib_count']).sort_index().plot.bar()


# # Text conversion

# # TFIDF

# In[ ]:


def vectorization(data, min_df, max_df):
    vectorizer = TfidfVectorizer(analyzer='word', stop_words='english', min_df=min_df, max_df=max_df)
    corpus = data
    return vectorizer.fit_transform(corpus).toarray(), vectorizer.get_feature_names()


# In[ ]:


X, feature_names = vectorization(articles['text'].values, 30, 200)
y = articles['claps']


# In[ ]:


Xdf = pd.DataFrame(X, columns=feature_names)

def normalized(col):
    return col.apply(lambda e: e/max(col))

Xdf['reading_time'] = normalized(articles['reading_time'])
Xdf['author_lib_count'] = normalized(articles['author_lib_count'])
Xdf


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(Xdf.values, y.values, test_size=0.3, random_state=145)
#X_train, X_test, y_train, y_test = train_test_split(Xdf[['reading_time', 'author_lib_count']], y.values, test_size=0.33, random_state=42)


# # Claps predictor

# In[ ]:


def compare(X_test, y_test, prediction):
    '''Comparison visualization function'''
    fig, ax = plt.subplots(figsize=(20, 10))
    width = 0.3
    x = np.arange(len(X_test))
    ax.bar(x - width/2, prediction(X_test), width, x, label='predicted')
    ax.bar(x + width/2, y_test, width, x, label='actual')
    ax.legend()
    fig.tight_layout()


# ## Linear Regression

# In[ ]:


reg = LinearRegression().fit(X_train, y_train)
print(f'Score: {reg.score(X_test, y_test)}')
compare(X_test, y_test, reg.predict)


# ## MLP

# In[ ]:


start = time()
mlp = MLPRegressor(hidden_layer_sizes=(1000), max_iter=400, learning_rate_init=0.01).fit(X_train, y_train)
print(f'done in {time() - start}s')
print(f'Score: {mlp.score(X_test, y_test)}')
compare(X_test, y_test, mlp.predict)


# \**Sad beginner sounds*\*

# # Article search

# ## Tag search

# In[ ]:


def rank(data, query, vectorization):
    pass
    # TODO


# In[ ]:


rank(articles, 'Artificial Intelligence', vectorization(articles['text'].values, 15, 150))

