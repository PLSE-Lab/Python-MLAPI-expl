#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


# In[ ]:


train_df = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
test_df = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
sub_df = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")


# In[ ]:


x = train_df["text"]
y = train_df["target"]


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# In[ ]:


vect = CountVectorizer(stop_words = 'english')

x_train_cv = vect.fit_transform(X_train)
x_test_cv = vect.transform(X_test)


# In[ ]:


X_train


# In[ ]:


clf = MultinomialNB()
clf.fit(x_train_cv, y_train)


# In[ ]:


pred = clf.predict(x_test_cv)


# In[ ]:


confusion_matrix(y_test, pred)


# In[ ]:


accuracy_score(y_test,pred)


# In[ ]:


y_test = test_df["text"]
y_test_cv = vect.transform(y_test)
preds = clf.predict(y_test_cv)


# In[ ]:


sub_df["target"] = preds
sub_df.to_csv("submission.csv",index=False)

