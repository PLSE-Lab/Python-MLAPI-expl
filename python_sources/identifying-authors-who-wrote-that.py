#!/usr/bin/env python
# coding: utf-8

# # Who wrote that? - Spooky author identification
# Started on 30 Oct 2017
# 
# This notebook is inspired by:
# * Machine Learning: Classification - Coursera course by University of Washington,
# https://www.coursera.org/learn/ml-classification
# * Machine Learning with Text in scikit-learn - Kevin Markham's tutorial at Pycon 2016, 
# https://m.youtube.com/watch?t=185s&v=ZiKMIuYidY0
# * Kernel by bshivanni - "Predict the author of the story", 
# https://www.kaggle.com/bsivavenu/predict-the-author-of-the-story
# * Kernel by SRK - "Simple Engg Feature Notebook - Spooky Author",
# https://www.kaggle.com/sudalairajkumar/simple-feature-engg-notebook-spooky-author

# Comments:
# 
# * In this kernel, I will also do a weighted averaging of the 'proba' of the 2 models to see the performance.
# 

# In[ ]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Read "train.csv" and "test.csv into pandas

# In[ ]:


train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')


# ## Examine the train data

# In[ ]:


train_df.shape


# In[ ]:


train_df.head()


# In[ ]:


# check the class distribution for the author label in train_df?
train_df['author'].value_counts()


# #### The class distribution looks balanced.

# In[ ]:


# compute the text length for the rows and record these
train_df['text_length'] = train_df['text'].apply(lambda x: len(str(x).split()))
train_df.head()


# In[ ]:


# look at the histogram plot for text length
train_df.hist()
plt.show()


# In[ ]:


# number of text length that are greater than 100, 150 & 200
G100 = sum(i > 100 for i in train_df['text_length'])
G150 = sum(i > 150 for i in train_df['text_length'])
G200 = sum(i > 200 for i in train_df['text_length'])
print('Text length greater than 100, 150 & 200 are ',G100,',',G150,'&',G200, ' respectively.')
print('In percentages, they are %.2f, %.2f & %.2f' %(G100/len(train_df)*100, 
      G150/len(train_df)*100, G200/len(train_df)*100))


# #### Most of the text length are within 200 words and less. Let's look at the summary statistics of the text lengths by author.

# In[ ]:


EAP = train_df[train_df['author'] =='EAP']['text_length']
EAP.describe()


# In[ ]:


EAP.hist()
plt.show()


# In[ ]:


MWS = train_df[train_df['author'] == 'MWS']['text_length']
MWS.describe()


# In[ ]:


MWS.hist()
plt.show()


# In[ ]:


HPL = train_df[train_df['author'] == 'HPL']['text_length']
HPL.describe()


# In[ ]:


HPL.hist()
plt.show()


# ## Similarly examine the text length & distribution in test data

# In[ ]:


test_df.shape


# In[ ]:


test_df.head()


# In[ ]:


# examine the text length in test_df and record these
test_df['text_length'] = test_df['text'].apply(lambda x: len(str(x).split()))
test_df.head()


# In[ ]:


test_df.hist()
plt.show()


# In[ ]:


# number of text length that are greater than 100, 150 & 200
G100 = sum(i > 100 for i in test_df['text_length'])
G150 = sum(i > 150 for i in test_df['text_length'])
G200 = sum(i > 200 for i in test_df['text_length'])
print('Text length greater than 100, 150 & 200 are ',G100,',',G150,'&',G200, ' respectively.')
print('In percentages, they are {:.2f}, {:.2f} & {:.2f}'.format(G100/len(test_df)*100, 
      G150/len(test_df)*100, G200/len(test_df)*100))


# #### The proportion of text which are long in the test data is very similar to that in the train data.

# ## Some preprocessing of the target variable to facilitate modelling

# In[ ]:


# convert author labels into numerical variables
train_df['author_num'] = train_df.author.map({'EAP':0, 'HPL':1, 'MWS':2})
# Check conversion for first 5 rows
train_df.head()


# ## Define X and y from train data for use in tokenization by CountVectorizer

# In[ ]:


X = train_df['text']
y = train_df['author_num']
print(X.shape)
print(y.shape)


# ## Split train data into a training and a test set

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[ ]:


# examine the class distribution in y_train and y_test
print(y_train.value_counts())
print(y_test.value_counts())


# ## Vectorize the data using CountVectorizer

# In[ ]:


# import and instantiate CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(lowercase=False, token_pattern=r'(?u)\b\w+\b|\,|\.|\;|\:')
# vect = CountVectorizer(lowercase=False, token_pattern=r'(?u)\b\w+\b|\,|\.|\?|\;|\:|\!|\'')
vect


# In[ ]:


# learn the vocabulary in the training data, then use it to create a document-term matrix
X_train_dtm = vect.fit_transform(X_train)
# examine the document-term matrix created from X_train
X_train_dtm


# In[ ]:


# transform the test data using the earlier fitted vocabulary, into a document-term matrix
X_test_dtm = vect.transform(X_test)
# examine the document-term matrix from X_test
X_test_dtm


# ## Build and evaluate an author classification model using Multinomial Naive Bayes

# In[ ]:


# import and instantiate the Multinomial Naive Bayes model
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb


# In[ ]:


# train the model using X_train_dtm & y_train
nb.fit(X_train_dtm, y_train)


# In[ ]:


# make author (class) predictions for X_test_dtm
y_pred_test = nb.predict(X_test_dtm)


# In[ ]:


# compute the accuracy of the predictions with y_test
from sklearn import metrics
metrics.accuracy_score(y_test, y_pred_test)


# In[ ]:


# compute the accuracy of training data predictions
y_pred_train = nb.predict(X_train_dtm)
metrics.accuracy_score(y_train, y_pred_train)


# In[ ]:


# look at the confusion matrix for y_test
metrics.confusion_matrix(y_test, y_pred_test)


# In[ ]:


# calculate predicted probabilities for X_test_dtm
y_pred_prob = nb.predict_proba(X_test_dtm)
y_pred_prob[:10]


# In[ ]:


# compute the log loss number
metrics.log_loss(y_test, y_pred_prob)


# ## Build and evaluate an author classification model using Logistic Regression

# In[ ]:


# import and instantiate the Logistic Regression model
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg


# In[ ]:


# train the model using X_train_dtm and y_train
logreg.fit(X_train_dtm, y_train)


# In[ ]:


# make class predictions for X_test_dtm
y_pred_test = logreg.predict(X_test_dtm)


# In[ ]:


# compute the accuracy of the predictions
metrics.accuracy_score(y_test, y_pred_test)


# In[ ]:


# compute the accuracy of predictions with the training data
y_pred_train = logreg.predict(X_train_dtm)
metrics.accuracy_score(y_train, y_pred_train)


# In[ ]:


# look at the confusion matrix for y_test
metrics.confusion_matrix(y_test, y_pred_test)


# In[ ]:


# compute the predicted probabilities for X_test_dtm
y_pred_prob = logreg.predict_proba(X_test_dtm)
y_pred_prob[:10]


# In[ ]:


# compute the log loss number
metrics.log_loss(y_test, y_pred_prob)


# ## Train the Logistic Regression model with the entire dataset from "train.csv"

# In[ ]:


# Learn the vocabulary in the entire training data, and create the document-term matrix
X_dtm = vect.fit_transform(X)
# Examine the document-term matrix created from X_train
X_dtm


# In[ ]:


# Train the Logistic Regression model using X_dtm & y
logreg.fit(X_dtm, y)


# In[ ]:


# Compute the accuracy of training data predictions
y_pred_train = logreg.predict(X_dtm)
metrics.accuracy_score(y, y_pred_train)


# ## Make predictions on the test data and compute the probabilities for submission

# In[ ]:


test_df.head()


# In[ ]:


# transform the test data using the earlier fitted vocabulary, into a document-term matrix
test_dtm = vect.transform(test_df['text'])
# examine the document-term matrix from X_test
test_dtm


# In[ ]:


# make author (class) predictions for test_dtm
LR_y_pred = logreg.predict(test_dtm)
print(LR_y_pred)


# In[ ]:


# calculate predicted probabilities for test_dtm
LR_y_pred_prob = logreg.predict_proba(test_dtm)
LR_y_pred_prob[:10]


# ## Train the Naive Bayes model with the entire dataset "train.csv"

# In[ ]:


nb.fit(X_dtm, y)


# In[ ]:


# compute the accuracy of training data predictions
y_pred_train = nb.predict(X_dtm)
metrics.accuracy_score(y, y_pred_train)


# ## Make predictions on test data

# In[ ]:


# make author (class) predictions for test_dtm
NB_y_pred = nb.predict(test_dtm)
print(NB_y_pred)


# In[ ]:


# calculate predicted probablilities for test_dtm
NB_y_pred_prob = nb.predict_proba(test_dtm)
NB_y_pred_prob[:10]


# ## Create submission file

# In[ ]:


alpha = 0.6
y_pred_prob = ((1-alpha)*LR_y_pred_prob + alpha*NB_y_pred_prob)
y_pred_prob[:10]


# In[ ]:


result = pd.DataFrame(y_pred_prob, columns=['EAP','HPL','MWS'])
result.insert(0, 'id', test_df['id'])
result.head()


# In[ ]:


# Generate submission file in csv format
result.to_csv('rhodium_submission_14.csv', index=False, float_format='%.20f')


# ### Will work on this further.
# ### Comments and tips are most welcomed.
# ### Please upvote if you find it useful. Cheers!

# In[ ]:




