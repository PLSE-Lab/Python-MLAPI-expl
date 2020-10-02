#!/usr/bin/env python
# coding: utf-8

# # Introduction 
# 
# I have created this dataset and notebook due to lack of data for fraudulent emails for supervised learning algorithms. I faced this issue when I was developing an approach for one of my projects. This notebook only contains necessary steps, not complete EDA. Feel free to leave your feedback and valuable comment.

# ## Import Data and Libraries 

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


# importing Natural Language Toolkit 
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score


# In[ ]:


df = pd.read_csv('../input/fraud_email_.csv')
df.head()


# In[ ]:


df.isnull().any()


# In[ ]:


df = df.dropna()


# In[ ]:


import nltk
nltk.download('stopwords')


# In[ ]:


stopset = set(stopwords.words("english"))
vectorizer = TfidfVectorizer(stop_words=stopset,binary=True)


# ## Test Train Split

# In[ ]:


# Extract feature column 'Text'
X = vectorizer.fit_transform(df.Text)
# Extract target column 'Class'
y = df.Class


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, train_size=0.80, random_state=42)


# ## Model

# In[ ]:


clf = RandomForestClassifier(n_estimators=15)


# In[ ]:


y_pred = clf.fit(X_train, y_train).predict_proba(X_test)


# In[ ]:


print(average_precision_score(y_test ,y_pred[:, 1]))

