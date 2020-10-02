#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

from matplotlib import pyplot as plt

import numpy as np # linear algebra
import pandas as pd # data processing


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


train.tail()


# In[ ]:


test.tail()


# In[ ]:


train = train.fillna(' ')
test = test.fillna(' ')


# In[ ]:


train['article'] = train['title'] + ' ' + train['author'] + ' ' + train['text']
test['article'] = test['title'] + ' ' + test['author'] + ' ' + test['text']


# In[ ]:


train.tail()


# In[ ]:


test.tail()


# In[ ]:


from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer


# In[ ]:


transformer = TfidfTransformer(smooth_idf=False)
ngram_vectorizer = CountVectorizer(ngram_range=(1, 2))


# In[ ]:


counts = ngram_vectorizer.fit_transform(train['article'].values)


# In[ ]:


tfidf = transformer.fit_transform(counts)


# In[ ]:


tfidf.data


# In[ ]:


targets = train['label'].values


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(tfidf, targets, random_state=0)


# In[ ]:


from sklearn.ensemble import AdaBoostClassifier


# In[ ]:


clf = AdaBoostClassifier(n_estimators=1)


# In[ ]:


clf.fit(X_train, y_train)


# In[ ]:


clf.score(X_train, y_train)


# In[ ]:


clf.score(X_test, y_test)

