#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# This advanced python project of detecting fake news deals with fake and real news. Using sklearn, we build a TfidfVectorizer on our dataset. Then, we initialize a PassiveAggressive Classifier and fit the model. In the end, the accuracy score and the confusion matrix tell us how well our model fares.

# # Loading Libraries

# In[ ]:


import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# # Loading Dataset

# In[ ]:


#Read the data
df=pd.read_csv('../input/news.csv')

#Get shape and head
print(df.shape)
df.head()


# In[ ]:


# Taking out the Target Variables
labels=df.label
labels.head()


# # Split the dataset

# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(df['text'], labels, test_size=0.2, random_state=7)


# # Analysis

# In[ ]:


# Initialize a TfidfVectorizer
tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)

# Fit and transform train set, transform test set
tfidf_train=tfidf_vectorizer.fit_transform(x_train) 
tfidf_test=tfidf_vectorizer.transform(x_test)


# In[ ]:


# Initialize a PassiveAggressiveClassifier
pac=PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)

# Predict on the test set and calculate accuracy
y_pred=pac.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')


# # Results

# In[ ]:


# Build confusion matrix
confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])


# # Refences:
# 1. [Detecting Fake News with Python](https://data-flair.training/blogs/advanced-python-project-detecting-fake-news/)
