#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv('/kaggle/input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv')


# In[ ]:


df.head()


# In[ ]:


len(df)


# In[ ]:


df['sentiment'].value_counts()


# In[ ]:


df.isnull().sum()


# In[ ]:


df['is_bad_review'] = df['sentiment'].apply(lambda x: 1 if x=='negative' else 0)


# In[ ]:


df.head()


# In[ ]:


blanks = []
for i, rv, se, is_bad in df.itertuples():
    if type(rv)==str:
        if rv.isspace():
            blanks.append(i)
print(len(blanks))


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X = df['review']
y = df['is_bad_review']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[ ]:


from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


# In[ ]:


text_clf_nb = Pipeline([('tfidf', TfidfVectorizer(stop_words = 'english')), ('clf', MultinomialNB())])
text_clf_lsvc = Pipeline([('tfidf', TfidfVectorizer(stop_words = 'english')), ('clf', LinearSVC())])


# In[ ]:


text_clf_nb.fit(X_train, y_train)


# In[ ]:


predictions_nb = text_clf_nb.predict(X_test)


# In[ ]:


from sklearn import metrics


# In[ ]:


print(metrics.confusion_matrix(y_test, predictions_nb))


# In[ ]:


print(metrics.classification_report(y_test, predictions_nb))


# In[ ]:


print(metrics.accuracy_score(y_test, predictions_nb))


# In[ ]:


text_clf_lsvc.fit(X_train, y_train)


# In[ ]:


predictions_lsvc = text_clf_lsvc.predict(X_test)


# In[ ]:


print(metrics.confusion_matrix(y_test, predictions_lsvc))


# In[ ]:


print(metrics.classification_report(y_test, predictions_lsvc))


# In[ ]:


print(metrics.accuracy_score(y_test, predictions_lsvc))

