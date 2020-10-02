#!/usr/bin/env python
# coding: utf-8

# ## O. Reference
# 
# [1]. [NLP Getting Started Tutorial](https://www.kaggle.com/philculliton/nlp-getting-started-tutorial)
# 
# [2]. [NLP with Disaster Tweets - EDA, Cleaning and BERT](https://www.kaggle.com/gunesevitan/nlp-with-disaster-tweets-eda-cleaning-and-bert#5.-Mislabeled-Samples)
# 
# This is a SVM version of [1] and I used random permutation to improve the model performance.
# For EDA part, [2] did excellent job on it.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn import svm

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# ## 1. Loading Data
# Use pd.read_csv() to load data from .csv files

# In[ ]:


train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')


# In[ ]:


train.head()


# In[ ]:


train.describe()


# ## 2. Missing Value

# In[ ]:


missing_cols = ['keyword', 'location']
plt.subplot(121)
plt.bar(train[missing_cols].isnull().sum().index, train[missing_cols].isnull().sum().values)
plt.title('Training Dataset')

plt.subplot(122)
plt.bar(test[missing_cols].isnull().sum().index, test[missing_cols].isnull().sum().values)
plt.title('Testing Dataset')


# In[ ]:


train[train['target'] == 0]['text'].values


# ## 3. Vector Converting
# Convert text to a vector using [sklearn.feature_extraction.text.CountVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)

# In[ ]:


vectorizer = CountVectorizer()
train_vec = vectorizer.fit_transform(train['text'])
test_vec = vectorizer.transform(test['text'])


# ## 4. Permutation

# In[ ]:


nsamp = train_vec.shape[0]
Iperm = np.random.permutation(nsamp)

xtr = train_vec[Iperm[:], :]
ytr = train['target']
ytr = ytr[Iperm[:]]


# ## 5. SVM model
# Implement [SVM classifier](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) to classify text.

# In[ ]:


clf = svm.SVC(C=2.6, verbose=10)


# In[ ]:


scores_perm = cross_val_score(clf, xtr, ytr, cv=3, scoring="f1")
scores = cross_val_score(clf, train_vec, train['target'], cv=3, scoring="f1")


# In[ ]:


print('Scores without permutation',scores)
print('Scores with permutation',scores_perm)


# ## 6. Fit model

# In[ ]:


clf.fit(xtr, ytr)


# ## 7. Prediction

# In[ ]:


sample_submission = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')


# In[ ]:


pred = clf.predict(test_vec)
sample_submission['target'] = pred


# In[ ]:


sample_submission.head()


# In[ ]:


sample_submission.to_csv("submission.csv", index=False)

