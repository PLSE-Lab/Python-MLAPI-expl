#!/usr/bin/env python
# coding: utf-8

# In[72]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import re
import time
import gc
import random

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[73]:


test_df = pd.read_csv('../input/test.csv')


# In[74]:


test_df.head()


# In[75]:


train_df = pd.read_csv('../input/train.csv')


# In[76]:


train_df.head()


# In[77]:


train_df.describe()


# In[78]:


train_df.info()


# In[79]:


train_df.groupby('target').qid.count()


# In[80]:


80810/1306122


# In[ ]:





# In[81]:




X_train, X_test, y_train, y_test = train_test_split(train_df['question_text'], 
                                                    train_df['target'], 
                                                    random_state=1)

print('Number of rows in the total set: {}'.format(train_df.shape[0]))
print('Number of rows in the training set: {}'.format(X_train.shape[0]))
print('Number of rows in the test set: {}'.format(X_test.shape[0]))


# In[82]:


# Instantiate the CountVectorizer method
count_vector = CountVectorizer()

# Fit the training data and then return the matrix
training_data = count_vector.fit_transform(X_train)

# Transform testing data and return the matrix. Note we are not fitting the testing data into the CountVectorizer()
testing_data = count_vector.transform(X_test)


# In[83]:


naive_bayes = MultinomialNB()
naive_bayes.fit(training_data, y_train)


# In[84]:


predictions = naive_bayes.predict(testing_data)


# In[85]:


print('Accuracy score: ', format(accuracy_score(y_test, predictions)))
print('Precision score: ', format(precision_score(y_test, predictions)))
print('Recall score: ', format(recall_score(y_test, predictions)))
print('F1 score: ', format(f1_score(y_test, predictions)))


# In[86]:


solution_data = count_vector.transform(test_df['question_text'])
Test_predictions = naive_bayes.predict(solution_data)


# In[ ]:





# In[87]:


df = pd.DataFrame(np.array(Test_predictions),columns = ['prediction'] ,index = test_df['qid'])


# In[88]:


df.to_csv('submission.csv', index = True)


# In[89]:


df.head()


# In[ ]:




