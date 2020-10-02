#!/usr/bin/env python
# coding: utf-8

# ## Predict which Tweets are about real disasters and which ones are not

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


train=pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
test=pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')


# In[ ]:


train.head()


# In[ ]:


X=train['text']
y=train['target']


# In[ ]:


train['target'].value_counts()


# In[ ]:


from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

text_clf = Pipeline([('tfidf', TfidfVectorizer(stop_words='english')),
                     ('clf', LinearSVC()),
])
text_clf.fit(X, y)
X_test=test['text']


# In[ ]:


test.head()


# In[ ]:


predictions = text_clf.predict(X_test)


# In[ ]:


predictions


# In[ ]:


sample_submission=pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')
sample_submission


# In[ ]:


submission = pd.DataFrame({
        "id": test["id"],
        "target": predictions
    }) 

filename = 'submission.csv'
submission.to_csv(filename,index=False)
print('Saved file: ' + filename)

