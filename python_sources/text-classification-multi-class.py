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


import nltk
from __future__ import division
import math
import string
import pandas as pd
import numpy as np


# In[ ]:


train=pd.read_csv("../input/train.tsv",delimiter='\t')


# In[ ]:


test=pd.read_csv("../input/test.tsv",delimiter='\t')


# In[ ]:


import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8,6))
train.groupby('Sentiment').Phrase.count().plot.bar(ylim=0)
plt.show()


# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
features = tfidf.fit_transform(train.Phrase).toarray()
labels = train.Sentiment
features.shape


# In[ ]:


from sklearn.svm import LinearSVC


# In[ ]:


model = LinearSVC()


# In[ ]:


model.fit(features, labels)


# In[ ]:


features_test = tfidf.transform(test.Phrase)

features_test.shape


# In[ ]:


pred = model.predict(features_test)


# In[ ]:


submission=pd.read_csv("../input/test.tsv",delimiter='\t')


# In[ ]:


submission['sentiment']=pred


# In[ ]:


submission.drop(['SentenceId', 'Phrase'], axis=1, inplace=True)


# In[ ]:


submission


# In[ ]:


submission.to_csv("submission.csv", index = False)


# In[ ]:




