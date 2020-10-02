#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from string import punctuation

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.svm import SVC

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ### Getting the data 

# Download data and look at the top 5 records

# In[ ]:


df = pd.read_csv(os.path.join('/', 'kaggle', 'input', 'bbc-fulltext-and-category', 'bbc-text.csv'))
df.head()


# In[ ]:


df.info()


# In[ ]:


df['category'].value_counts()


# In[ ]:


df['category'].value_counts().plot.bar(figsize=(12, 8))


# Encode categories into labels

# In[ ]:


le = LabelEncoder()
df['labels'] = le.fit_transform(df['category'])
df.head()


# Tokenize texts, stem and remove punctuation and stopwords

# In[ ]:


stemmer = PorterStemmer()
to_exclude = set(stopwords.words('english') + list(punctuation))
filter_func = lambda text: ' '.join([stemmer.stem(word) for word in word_tokenize(text) if word not in to_exclude])


# In[ ]:


df['text'] = df['text'].apply(filter_func)
df.head()


# Using TF-IDF vectorizer including 2-grams

# In[ ]:


tf_vect = TfidfVectorizer(ngram_range=(1, 2))
X = tf_vect.fit_transform(df['text'].values)
y = df['labels'].values

print('X shape: {}, y shape: {}'.format(X.shape, y.shape))


# We will use SVM and find best kernel by GridSearch

# In[ ]:


svc = SVC(random_state=1)
svc.get_params()


# Define grid

# In[ ]:


svc_grid = {
    'kernel': ['poly', 'rbf', 'sigmoid', 'linear'],
    'C': np.linspace(.1, .9, 6)
}


# In[ ]:


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
grid_search_estimator = GridSearchCV(svc, svc_grid, scoring='accuracy', cv=cv, n_jobs=-1)
grid_search_estimator.fit(X, y)


# Look at the best params and score

# In[ ]:


grid_search_estimator.best_params_


# In[ ]:


print('SVM best score: {:.8f}'.format(grid_search_estimator.best_score_))


# Confusion matrix and classification report(precision, recall, F1-score)

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
predicted = grid_search_estimator.best_estimator_.fit(X_train, y_train).predict(X_test)


# In[ ]:


print(classification_report(y_test, predicted))


# In[ ]:


print(confusion_matrix(y_test, predicted))

