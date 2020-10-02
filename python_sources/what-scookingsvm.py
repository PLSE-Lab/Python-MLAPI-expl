#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd
train = pd.read_json("../input/train.json")
test = pd.read_json("../input/test.json")

print (train.head())

ytrain = train.cuisine
print (ytrain.head(5))

Id = test.id
print (Id.head(5))


# In[ ]:


train.cuisine.value_counts().plot(kind='bar')


# In[ ]:


def arraytotext(records): return [" ".join(record).lower() for record in records]


# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[ ]:


tfidf = TfidfVectorizer(binary=True)


# In[ ]:


train_tfidf_features = tfidf.fit_transform(arraytotext(train.ingredients))
test_tfidf_features= tfidf.transform(arraytotext(test.ingredients))


# In[ ]:


from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score


# In[ ]:


classifier = SVC(C=200, kernel='rbf', degree=3,gamma=1, coef0=1, shrinking=True,tol=0.001, probability=False, cache_size=200,class_weight=None, verbose=False, max_iter=-1,decision_function_shape=None,random_state=None)


# In[ ]:


model = OneVsRestClassifier(classifier)


# In[ ]:


scores = cross_val_score(classifier,train_tfidf_features, ytrain, cv=5)
print ("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# In[ ]:


model.fit(train_tfidf_features, ytrain)


# In[ ]:


predictions = model.predict(test_tfidf_features)
print (predictions)


# In[ ]:


submission = pd.DataFrame()
submission['id'] = Id
submission['cuisine'] = predictions
submission.to_csv('submissionSVM.csv', index=False)

