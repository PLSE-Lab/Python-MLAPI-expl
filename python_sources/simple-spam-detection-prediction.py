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


df=pd.read_csv('../input/sms-spam-collection-dataset/spam.csv',encoding='latin')
df.head()


# In[ ]:


df=df[['v1','v2']]
df.head()


# In[ ]:


df.columns=['label','sms']
df.isnull().sum()


# In[ ]:


df['label'].value_counts()


# In[ ]:


from sklearn.model_selection import train_test_split


X=df['sms']
y=df['label']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)


# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer=TfidfVectorizer()
X_train_tfidf=vectorizer.fit_transform(X_train)
X_train_tfidf.shape


# In[ ]:


from sklearn.svm import LinearSVC
clf=LinearSVC()
clf.fit(X_train_tfidf,y_train)


# In[ ]:


from sklearn.pipeline import Pipeline


text_clf = Pipeline([('tfidf', TfidfVectorizer()),
                     ('clf', LinearSVC()),
])

# Feed the training data through the pipeline
text_clf.fit(X_train, y_train)


# 

# In[ ]:


predictions=text_clf.predict(X_test)


# In[ ]:


from sklearn import metrics
metrics.confusion_matrix(y_test,predictions)


# In[ ]:


print(metrics.classification_report(y_test,predictions))


# In[ ]:


print(metrics.accuracy_score(y_test,predictions))

