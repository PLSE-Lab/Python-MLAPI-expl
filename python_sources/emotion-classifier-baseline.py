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
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_context("talk")
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ## Reading Data:

# In[ ]:


data = pd.read_csv('/kaggle/input/emotion.data')
data.info()


# In[ ]:


data.head()


# In[ ]:


data['emotions'].value_counts().plot.bar()
plt.show()


# In[ ]:


data = data.drop(['Unnamed: 0'], axis=1)
X = data['text']
y = data['emotions']


# ## Data Modeling:

# In[ ]:


train, test = train_test_split(data, test_size = 0.2, random_state = 12, stratify = data['emotions'])


# In[ ]:


X_train = train.text
y_train = train.emotions
X_test = test.text
y_test = test.emotions


# In[ ]:


vectorizer = TfidfVectorizer( max_df= 0.9).fit(X_train)
X_train = vectorizer.transform(X_train)
X_test = vectorizer.transform(X_test)
print(X_train.shape)


# In[ ]:


encoder = LabelEncoder().fit(y_train)
y_train = encoder.transform(y_train)
y_test = encoder.transform(y_test)


# ## Model Training:

# In[ ]:


model = LogisticRegression(C=.1, class_weight='balanced')
model.fit(X_train, y_train)
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)
print("Training Accuracy : ", accuracy_score(y_train, y_pred_train))
print("Testing Accuracy  : ", accuracy_score(y_test, y_pred_test))


# ## Predictions:

# In[ ]:


def predict_(x, plot=False):
    tfidf = vectorizer.transform([x])
    preds = model.predict_proba(tfidf)[0]
    plt.figure(figsize=(8,4))
    sns.barplot(x= encoder.classes_, y=preds)
    plt.show()
    return preds


# In[ ]:


text = "this kernel gives a baseline LR model for the problem fairly well, although we can improve it"


# In[ ]:


predict_(text, plot=True)

