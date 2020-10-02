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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier


# In[ ]:


train = pd.read_table('../input/train.tsv')
test = pd.read_table('../input/test.tsv')


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


xtrain, xtest, ytrain, ytest = train_test_split(train.Phrase, train.Sentiment, test_size = 0.2, random_state = 42)


# In[ ]:


cv = CountVectorizer(max_features = None)
cv.fit(xtrain)


# In[ ]:


xtrain_cv = cv.transform(xtrain)
xtrain_cv


# In[ ]:


xtest_cv = cv.transform(xtest)
xtest_cv


# In[ ]:


test_cv = cv.transform(test)
test_cv


# In[ ]:


etc = ExtraTreesClassifier()
etc.fit(xtrain_cv,ytrain)
pred = etc.predict(test_cv)


# In[ ]:


pred


# In[ ]:


print('Accuracy of Extra Tree Classifier: ', etc.score( xtrain_cv , ytrain))

