#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# New Kernel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
import os
print(os.listdir("../input"))


# In[ ]:


train = pd.read_csv('../input/labeledTrainData.tsv',header = 0, delimiter = '\t')
test = pd.read_csv('../input/testData.tsv',header = 0, delimiter = '\t')


# In[ ]:


reviews = train['review']
sentiments = train['sentiment']


# In[ ]:


reviewsT = test['review']


# In[ ]:


word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='ascii',
    analyzer='word',
    stop_words='english',
    ngram_range=(1, 1),
    max_features=10000)
word_vectorizer.fit(reviews + reviewsT)


# In[ ]:


X = word_vectorizer.transform(reviews)
X.shape


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, sentiments,test_size=0.25)
print (X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# In[ ]:


model=LogisticRegression(solver='lbfgs')
model.fit(X_train,y_train)


# In[ ]:


preds=model.predict(X_test)
print(classification_report(preds,y_test))


# In[ ]:


Xt = word_vectorizer.transform(reviewsT)
Xt.shape


# In[ ]:


preds=model.predict(Xt)
test['sentiment'] = preds


# In[ ]:


test


# In[ ]:


pd.read_csv('../input/sampleSubmission.csv')


# In[ ]:


test = test.drop(['review'],axis = 1)


# In[ ]:


test.to_csv('submissions.csv',index=False)


# In[ ]:




