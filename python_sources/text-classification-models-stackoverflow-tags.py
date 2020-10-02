#!/usr/bin/env python
# coding: utf-8

# # Text Classification Models
# 
# The intention of this notebook is to look the below five models commonly used for text classification:
# * Naive Bayes Classifier
# * Linear Support Vector Machines
# * Linear SVM with SGD
# * Logistic Classifier
# * RandomForest Classifier
# 
# The dataset involves StackOverflow posts and the tags. The goal will be to predict the tags based on the contents of the posts.
# 
# 

# #### Importing Relevant Libraries

# In[ ]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import classification_report


# #### Reading Data

# In[ ]:


df = pd.read_csv('/kaggle/input/stack-overflow-data.csv')
df = df[pd.notnull(df['tags'])]


# #### Pre-processing

# In[ ]:


def clean_text(text):
    """
        text: a string
        return: modified initial string
    """
    text = BeautifulSoup(text, "lxml").text # HTML decoding
    text = text.lower() # lowercase text
    return text

df['post'] = df['post'].apply(clean_text)


# In[ ]:


df.head()


# #### Train-Test Split

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(df['post'], df['tags'], random_state=1087, test_size=0.2)


# ## Classification Models
# 
# 
# ### 1) Naive Bayes Classifier: 

# In[ ]:


nb = Pipeline([
                ('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', MultinomialNB()),
              ])

nb.fit(X_train, y_train)

y_pred = nb.predict(X_test)

print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred))


# ### 2) Linear SVM Classifier:

# In[ ]:


svm = Pipeline([
                ('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', LinearSVC()),
              ])

svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)

print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred))


# ### 3) Linear SVM with SGD:

# In[ ]:


svm_sgd = Pipeline([
                ('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, max_iter=10, random_state=42))
              ])

svm_sgd.fit(X_train, y_train)

y_pred = svm_sgd.predict(X_test)

print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred))


# ### 4) Logistic Classifier:

# In[ ]:


lgclf = Pipeline([
                ('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', LogisticRegression(random_state=0)),
              ])

lgclf.fit(X_train, y_train)

y_pred = lgclf.predict(X_test)

print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred))


# ### 5) RandomForest Classifier:

# In[ ]:


rfc = Pipeline([
                ('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', RandomForestClassifier(n_estimators=200, max_depth=5, random_state=0)),
              ])

rfc.fit(X_train, y_train)

y_pred = rfc.predict(X_test)

print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred))


# Report:
# 
#                      Model      Accuracy
#     1) Naive Bayes Classifier   74.6%
#     2) Linear SVM               81.1%
#     3) Linear SVM with SGD      79.1%
#     4) Logistic Classifier      81.0%
#     5) RandomForest Classifier  74.1%
# 
# This completes this kernel.

# In[ ]:




